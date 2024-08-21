# %%
from OpInf import *
import numpy as np
import sys
import scipy
import h5py
import tqdm
import json
import umap
import torch
import adios4dolfinx
import dolfinx
import cProfile
import pstats
import io
import numpy as np
import h5py
from scipy import linalg
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from dolfinx import fem, mesh, io, plot

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from itertools import product, combinations

from sklearn.manifold import SpectralEmbedding

from pathlib import Path
import json
import stiefel
import pickle as pkl
from stiefel import *
import importlib

importlib.reload(stiefel)

import importlib

importlib.reload(stiefel)

from scipy.io import loadmat
from operators import ckron, ckron_indices

import scienceplots

plt.style.use(["science", "grid"])
plt.rcParams.update({"font.size": 16})

# set numpy random seed
np.random.seed(42)
from Representation import *


import tqdm


def initialize_hdf5_store(store_path, shape, N_samples):
    with h5py.File(store_path, "w") as f:
        f.create_dataset("s_rec_full", shape=(N_samples,) + shape, dtype=np.float32)
        f.create_dataset("valid_mask", shape=(N_samples,), dtype=bool)
    return store_path


def save_s_rec_full_batch(store_path, s_rec_full_batch, start_idx, expected_shape):
    valid_samples = []
    valid_indices = []
    for i, sample in enumerate(s_rec_full_batch):
        if sample.shape == expected_shape:
            valid_samples.append(sample)
            valid_indices.append(start_idx + i)
        else:
            print(
                f"Skipping sample at index {start_idx + i} due to inconsistent shape. Expected {expected_shape}, got {sample.shape}"
            )

    num_valid = len(valid_samples)
    if num_valid > 0:
        try:
            with h5py.File(store_path, "r+") as f:
                for i, sample in zip(valid_indices, valid_samples):
                    f["s_rec_full"][i] = sample
                    f["valid_mask"][i] = True
            print(f"Successfully saved {num_valid} samples.")
        except Exception as e:
            print(f"Error occurred while saving batch: {str(e)}")
            num_valid = 0  # Reset to 0 if save failed

    return num_valid


# -------------------------------

# Load data
t_start = 4
T_end_train = 5

variable = "p"

# Load the HDF5 file
combined_dataseet_filepath = Path(
    f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_combined_dataset.h5"
)

with h5py.File(combined_dataseet_filepath, "r") as f:
    mus = f.attrs["mu_values"]
    U_c = f.attrs["U_c"]
    L_c = f.attrs["L_c"]
    rho = f.attrs["rho"]

    # Get the shape of the data
    sample_group = f[f"mu_{mus[0]:.6f}"]
    data_shape = sample_group["snapshots"].shape


test_mu = 0.000556
test_Re = int(np.round(1.0 * 0.1 * 1 / test_mu))
X_all_test = np.load(
    f"/data1/jy384/research/Data/UnimodalSROB/ns/Re_{test_Re}_mu_{test_mu}/{variable}_snapshots_matrix_test.npy"
).T

import yaml
from omegaconf import OmegaConf

base_path = Path("/data1/jy384/research/Data/UnimodalSROB/ns/")
effective_dts = {}

for mu in mus:
    Re = int(
        np.round(1.0 * 0.1 * 1 / mu)
    )  # Calculate Re based on U_c, L_c, rho, and mu
    folder_name = f"Re_{Re}_mu_{mu:.6f}"
    metadata_path = (
        base_path / folder_name / f"{variable}_snapshots_matrix_train_metadata.yaml"
    )

    print("Metadata path: ", metadata_path)

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            config_dict = yaml.safe_load(f)

        cfg = OmegaConf.create(config_dict)
        effective_dts[mu] = cfg.effective_dt
    else:
        print(f"Metadata file not found for mu = {mu}")

for mu, dt in effective_dts.items():
    print(f"For mu = {mu:.6f}, effective dt = {dt}")


p = 2
Mp = len(mus)

dt = effective_dts[mus[0]]

config = {
    "N": data_shape[0],
    "dt": 1e-3,
    "T_end": T_end_train,
    "mus": list(mus),
    "Mp": Mp,
    "K": T_end_train / dt,  # T_end / dt
    "DS": 1,
    "params": {
        "modelform": "LQCP",
        "modeltime": "continuous",
        "dt": dt,
        "ddt_order": "4c",
        "p": p,  # polynomial order
        "lambda1": 5e-2,
        "lambda2": 5e-2,
        "lambda3": 100,
    },
    "type": "single",
}


# %%
# a colors list to tag the group of combinations
colors = plt.get_cmap("tab20").colors

# Create a dictionary to store combinations with their respective colors
tagged_combinations = {}

for r in range(1, len(mus) - 1):
    combinations_ = list(combinations(mus[1:-1], r))
    combinations_ = [(mus[0],) + combo + (mus[-1],) for combo in combinations_]
    print(f"{len(combinations_)} of combinations generated: ", combinations_)

    # Ensure that there are enough colors, cycle through colors if necessary
    color = colors[(r - 1) % len(colors)]

    # Store the combinations with their tag
    tagged_combinations[color] = combinations_

# Flatten the list and maintain the color tagging
flattened_tagged_combinations = [
    (item, color) for color, combos in tagged_combinations.items() for item in combos
]

# Example of accessing the flattened list with color tags
for combo, color in flattened_tagged_combinations:
    print(f"Combination: {combo}, Color: {color}")

# randomly draw 3 ICs (mus) without replacement from X_all_nominal
drawn_mus = [tup[0] for tup in flattened_tagged_combinations]
color_tags = []
for n_X in range(len(flattened_tagged_combinations)):
    mus_list = mus.tolist()
    mus_idx = [mus_list.index(mus_) for mus_ in flattened_tagged_combinations[n_X][0]]
    # print(mus_idx)
    color_tags.append(flattened_tagged_combinations[n_X][1])

# X_list.append(X_nominal)
# color_tags.append((0,1,1)) # cyan

rob_lst = []
rel_err_SVD_lst = []
idx_lst = []
# names = [f"tap={taps}" for taps in numtaps] + ["Nominal"]
names = [f"$\mu$={mus}" for mus in drawn_mus]

fig, ax = plt.subplots(figsize=(8, 6))

err_tol = 5e-2


# %%
# Model parameters

tol = 1e-3  # tolerence for alternating minimization
gamma = 0.01  # regularization parameter\
max_iter = 100  # maximum number of iterations

import os


with open(base_path / f"{variable}_representatives_data_KMeans.pkl", "rb") as f:
    representatives_data = pkl.load(f)

representatives = representatives_data["indices"]
n_clusters = representatives_data["n_clusters"]
cluster_labels = representatives_data["cluster_labels"]

X_list_sel = []
with h5py.File(combined_dataseet_filepath, "r") as file:
    for i, index in tqdm.tqdm(enumerate(representatives)):
        # i = i_ + 11
        X = []
        combo = drawn_mus[index]
        for mu in combo:
            grp = file[f"mu_{mu:.6f}"]
            snapshot = grp["snapshots"][:]
            X.append(snapshot)

        X = np.concatenate(X, axis=1)

        X_list_sel.append(X)


basis_operators_file = base_path / f"{variable}_basis_operators_mu_{test_mu:.6f}.hdf5"
print("Checking if file at", basis_operators_file, "exists")
if basis_operators_file.exists():
    print("Loading selected operators from file")
    with h5py.File(basis_operators_file, "r") as f:
        # check if number of operators is the same as number of combos
        if len(f["operators_lst"]) != len(drawn_mus):
            raise ValueError(
                f"Number of operators in file {basis_operators_file} is not the same as number of combos"
            )
        operators_lst_sel = []

        Vr_lst = f["Vr_lst"][:]
        Vbar_lst = f["Vbar_lst"][:]

        Vr_lst_sel = []
        Vbar_lst_sel = []

        # Load operators_lst
        operators_lst_sel = []
        operators_grp = f["operators_lst"]
        for index in representatives:
            ops = {}
            combo_grp = operators_grp[f"combo_{index}"]
            for key in combo_grp.keys():
                ops[key] = combo_grp[key][:]
            operators_lst_sel.append(ops)

            Vr_lst_sel.append(Vr_lst[index])
            Vbar_lst_sel.append(Vbar_lst[index])

        print("Loaded basis and operators data")

        # Now you can use Vr_lst, Vbar_lst, and operators_lst in your code
        print(f"Number of Vr matrices: {len(Vr_lst)}")
        print(f"Number of Vbar matrices: {len(Vbar_lst)}")
        print(f"Number of selected operator sets: {len(operators_lst_sel)}")
        print("Shape of Vr matrix: ", Vr_lst[0].shape)
        print("Shape of Vbar matrix: ", Vbar_lst[0].shape)


r = Vr_lst[0].shape[1]
q_trunc = Vbar_lst[0].shape[1]


V_combined_lst = [
    np.concatenate([Vr, Vbar], axis=1) for Vr, Vbar in zip(Vr_lst, Vbar_lst)
]

X_all_global = np.concatenate(X_list_sel, axis=1)
X_all_global = X_all_global - np.mean(X_all_global, axis=1)[:, None]

V_combined_global = np.linalg.svd(X_all_global, full_matrices=False)[0][
    :, : r + q_trunc
]

V_combined1 = V_combined_global
Vr1 = V_combined1[:, :r]
Vbar1 = V_combined1[:, r:]

for idx in range(len(Vr_lst)):
    Vr_idx = Vr_lst[idx]
    Vbar_idx = Vbar_lst[idx]

    for j in range(Vr_idx.shape[1]):
        dist1 = np.linalg.norm(Vr1[:, j] - Vr_idx[:, j])
        dist2 = np.linalg.norm(Vr1[:, j] + Vr_idx[:, j])
        if dist2 < dist1:
            Vr_lst[idx][:, j] = -Vr_lst[idx][:, j]

    for j in range(Vbar_idx.shape[1]):
        dist1 = np.linalg.norm(Vbar1[:, j] - Vbar_idx[:, j])
        dist2 = np.linalg.norm(Vbar1[:, j] + Vbar_idx[:, j])
        if dist2 < dist1:
            Vbar_lst[idx][:, j] = -Vbar_lst[idx][:, j]

    for j in range(V_combined1.shape[1]):
        dist1 = np.linalg.norm(V_combined1[:, j] - V_combined_lst[idx][:, j])
        dist2 = np.linalg.norm(V_combined1[:, j] + V_combined_lst[idx][:, j])
        if dist2 < dist1:
            V_combined_lst[idx][:, j] = -V_combined_lst[idx][:, j]


# In your main loop
store_path = f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_ReconStates.h5"
sample_shape = X_all_test.shape
N_samples = 1000

store_path = initialize_hdf5_store(store_path, sample_shape, N_samples)

batch_size = 20
s_rec_full_batch = []

stiefel_samples_file = base_path / f"{variable}_stiefel_samples_combined_KMeans.h5"
basis_file = f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_basis_r{r}_q{q_trunc}_p{p}_gamma{gamma}.h5"

Xi_lst_sel = []
with h5py.File(basis_file, "r") as basis_f:
    print("Loading selected Xi_lst")
    for i, index in tqdm.tqdm(enumerate(representatives)):
        grp = basis_f[f"combo_{index}"]
        Xi_lst_sel.append(grp["Xi"][:])


total_valid_samples = 0
with h5py.File(stiefel_samples_file, "r") as stiefel_file:
    stiefel_samples = stiefel_file["stiefel_samples"][:]
    maxI = stiefel_file["maxI"][:]
    for i in tqdm.tqdm(range(N_samples)):
        # ... your existing code to generate s_rec_full ...
        X = X_list_sel[maxI[i]]
        X_ref = np.mean(X, axis=1)[:, None]

        Vr_sample = stiefel_samples[i][:, :r]
        Vbar_sample = stiefel_samples[i][:, r : r + q_trunc]

        operators = operators_lst_sel[maxI[i]]

        q0 = Vr_sample.T @ (X_all_test[:, 0][:, None] - X_ref).flatten()

        T_end_full = 8
        time_domain_full = np.arange(t_start, T_end_full, dt)

        multi_indices = generate_multi_indices_efficient(len(q0), p=p)

        modelform = config["params"]["modelform"]
        out_full = scipy.integrate.solve_ivp(
            rhs,  # Integrate this function
            [time_domain_full[0], time_domain_full[-1]],  # over this time interval
            q0,  # from this initial condition
            t_eval=time_domain_full,  # evaluated at these points
            args=[
                operators,
                config["params"],
                None,
                multi_indices,
            ],  # additional arguments to rhs
        )

        s_hat_full = out_full.y
        poly_full = np.concatenate(polynomial_form(s_hat_full, p=p), axis=0)
        Xi = Xi_lst_sel[maxI[i]]

        s_rec_full = X_ref + Vr_sample @ s_hat_full + Vbar_sample @ Xi @ poly_full

        s_rec_full_batch.append(s_rec_full)

        if len(s_rec_full_batch) == batch_size or i == N_samples - 1:
            valid_samples = save_s_rec_full_batch(
                store_path,
                s_rec_full_batch,
                i - len(s_rec_full_batch) + 1,
                sample_shape,
            )
            total_valid_samples += valid_samples
            s_rec_full_batch = []

print(f"Saving completed. Total valid samples: {total_valid_samples}")
with h5py.File(store_path, "r") as f:
    print(f"Valid samples according to mask: {np.sum(f['valid_mask'][:])}")
