# %%
import dolfinx
import dolfinx.fem.petsc

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
np.random.seed(3)


# %%
from Representation import *

from basix.ufl import element
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_topological,
    set_bc,
    Expression,
)

from dolfinx.io import VTXWriter, distribute_entity_data, gmshio

# %%
# mus = [1.1, 1.05, 1, 0.95, 0.9]
# mus = [1.15, 1.1, 1.05, 1, 0.95, 0.9, 0.85]
# mus = [0.4, 0.6, 0.8, 1.0, 1.2]
mus = [
    0.002000,
    0.001333,
    0.001000,
    #    -----
    0.001111,
    #    -----
    0.000909,
    #    -----
    0.000800,
    0.000667,
    0.000571,
    0.000500,
]

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
# p = 3

tol = 1e-3  # tolerence for alternating minimization
gamma = 0.01  # regularization parameter\
max_iter = 100  # maximum number of iterations

import os

basis_operators_file = base_path / f"{variable}_basis_operators_mu_{test_mu:.6f}.hdf5"
print("Checking if file at", basis_operators_file, "exists")
if basis_operators_file.exists():
    print("Loading operators from file")
    with h5py.File(basis_operators_file, "r") as f:
        # check if number of operators is the same as number of combos
        if len(f["operators_lst"]) != len(drawn_mus):
            raise ValueError(
                f"Number of operators in file {basis_operators_file} is not the same as number of combos"
            )
        operators_lst = []

        Vr_lst = f["Vr_lst"][:]
        Vbar_lst = f["Vbar_lst"][:]

        # Load operators_lst
        operators_lst = []
        operators_grp = f["operators_lst"]
        for i in range(len(operators_grp)):
            ops = {}
            combo_grp = operators_grp[f"combo_{i}"]
            for key in combo_grp.keys():
                ops[key] = combo_grp[key][:]
            operators_lst.append(ops)

        print("Loaded basis and operators data")

        # Now you can use Vr_lst, Vbar_lst, and operators_lst in your code
        print(f"Number of Vr matrices: {len(Vr_lst)}")
        print(f"Number of Vbar matrices: {len(Vbar_lst)}")
        print(f"Number of operator sets: {len(operators_lst)}")


r = Vr_lst[-1].shape[1]
q_trunc = Vbar_lst[-1].shape[1]


# Aligning the signs of Vr_lst with Vr_lst[-1]
Vr1 = Vr_lst[-1]
for idx in range(len(Vr_lst) - 1):
    Vr_idx = Vr_lst[idx]
    for j in range(Vr_idx.shape[1]):
        dist1 = np.linalg.norm(Vr1[:, j] - Vr_idx[:, j])
        dist2 = np.linalg.norm(Vr1[:, j] + Vr_idx[:, j])
        if dist2 < dist1:
            Vr_lst[idx][:, j] = -Vr_lst[idx][:, j]


# For Vbar_lst, ensuring sign agreement with Vbar_lst[-1]
Vbar1 = Vbar_lst[-1]
for idx in range(len(Vbar_lst) - 1):
    Vbar_idx = Vbar_lst[idx]
    for j in range(Vbar_idx.shape[1]):
        dist1 = np.linalg.norm(Vbar1[:, j] - Vbar_idx[:, j])
        dist2 = np.linalg.norm(Vbar1[:, j] + Vbar_idx[:, j])
        if dist2 < dist1:
            Vbar_lst[idx][:, j] = -Vbar_lst[idx][:, j]

V_combined_lst = [
    np.concatenate([Vr, Vbar], axis=1) for Vr, Vbar in zip(Vr_lst, Vbar_lst)
]

# plot spectral embedding of the generated stiefel sampels
flat_Vr_arr = np.concatenate([rob.flatten()[:, None] for rob in Vr_lst], axis=1)
flat_Vbar_arr = np.concatenate([rob.flatten()[:, None] for rob in Vbar_lst], axis=1)
flat_V_combined_arr = np.concatenate(
    [rob.flatten()[:, None] for rob in V_combined_lst], axis=1
)

# colors =
colors = plt.cm.tab20c(np.linspace(0, 1, len(Vr_lst)))


# Perform spectral embedding
embedder = SpectralEmbedding(n_components=2)
embedding_Vr = embedder.fit_transform(flat_Vr_arr.T)
embedding_Vbar = embedder.fit_transform(flat_Vbar_arr.T)

T_end_index = int((T_end_train - t_start) / dt)


# %%
def pairwise_mat_distances(matrices, norm_function):
    num_matrices = len(matrices)

    # Initialize a 2D array with zeros
    pair_dist = np.zeros((num_matrices, num_matrices))

    for i in range(num_matrices):
        for j in range(i + 1, num_matrices):
            matrix_a = matrices[i]
            matrix_b = matrices[j]
            norm_diff = norm_function(matrix_a, matrix_b)
            # Store the norm difference in the pair_dist array
            pair_dist[i][j] = norm_diff
            pair_dist[j][i] = norm_diff  # Since distance is symmetric

    # normalize the distance matrix
    max_val = np.max(pair_dist)
    min_val = np.min(pair_dist)
    pair_dist = (pair_dist - min_val) / (max_val - min_val)
    return pair_dist


def frobenius_norm_difference(matrix_a, matrix_b):
    norm_a = np.linalg.norm(matrix_a, "fro")
    norm_b = np.linalg.norm(matrix_b, "fro")
    diff_norm = np.linalg.norm(matrix_a - matrix_b, "fro")
    return diff_norm


# %%
# calculate the normalized frobenius norm difference between the V_combined basis
V_combined_frobenius = pairwise_mat_distances(V_combined_lst, frobenius_norm_difference)

# plot the spectral embedding of the V_combined basis colored by the normalized frobenius norm difference
V_combined_embedding = embedder.fit_transform(flat_V_combined_arr.T)

# %%
with open(base_path / f"{variable}_representatives_data_KMeans.pkl", "rb") as f:
    representatives_data = pkl.load(f)

representatives = representatives_data["indices"]
n_clusters = representatives_data["n_clusters"]
cluster_labels = representatives_data["cluster_labels"]


def compute_magnitude(s_rec):
    """
    Compute the magnitude of the reconstructed state.

    :param s_rec: Reconstructed state array of shape (15168, 800)
    :return: Magnitude array of shape (7584, 800)
    """
    # interleaved u and v components
    component_len = s_rec.shape[0] // 2
    u = s_rec[0::2, :]
    v = s_rec[1::2, :]

    # Compute magnitude
    magnitude = np.sqrt(u**2 + v**2)

    return magnitude


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

# %%
X_all_global = np.concatenate(X_list_sel, axis=1)
X_all_global = X_all_global - np.mean(X_all_global, axis=1)[:, None]

# %%
names_sel = [names[i] for i in representatives]
names_sel.append("Global")

# %%
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

Vr_lst_sel = []
Vbar_lst_sel = []
V_combined_lst_stochastic = []

for i in representatives:
    V_combined_lst_stochastic.append(V_combined_lst[i])
    Vr_lst_sel.append(Vr_lst[i])
    Vbar_lst_sel.append(Vbar_lst[i])

V_combined_lst_stochastic.append(V_combined_global)
Vr_lst_sel.append(V_combined_global[:, :r])
Vbar_lst_sel.append(V_combined_global[:, r:])

assert np.allclose(V_combined_lst_stochastic[0][:, :r], Vr_lst_sel[0])

# %%
s_rec_full_file = base_path / f"{variable}_s_rec_full_mu_{test_mu:.6f}_KMeans.h5"
with h5py.File(s_rec_full_file, "r") as s_rec_file:
    # Load a specific combo's s_rec_full
    combo_index = representatives[0]  # or any other index
    s_rec_full_0 = s_rec_file[f"combo_{combo_index}"][:]

    # Or load all combos
    # all_s_rec_full = {key: s_rec_file[key][:] for key in s_rec_file.keys()}

print("Loaded s_rec_full data")

# %%
s_rec_full_0 = compute_magnitude(s_rec_full_0)


# %%
# read mesh
def print_mesh_info(mesh: dolfinx.mesh.Mesh):
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    node_map = mesh.geometry.index_map()
    print(
        f"Rank {mesh.comm.rank}: number of owned cells {cell_map.size_local}",
        f", number of ghosted cells {cell_map.num_ghosts}\n",
        f"Number of owned nodes {node_map.size_local}",
        f", number of ghosted nodes {node_map.num_ghosts}",
    )


def read_mesh(filename: Path):
    from mpi4py import MPI

    import dolfinx

    import adios4dolfinx

    mesh = adios4dolfinx.read_mesh(
        filename,
        comm=MPI.COMM_WORLD,
        engine="BP4",
        ghost_mode=dolfinx.mesh.GhostMode.none,
    )
    print_mesh_info(mesh)

    return mesh


# %%
mesh_path = (
    base_path / f"Re_{test_Re}_mu_{test_mu}" / f"{variable}_snapshots_mu{test_mu}.bp"
)
mymesh = read_mesh(mesh_path)


N_stiefel_samples = 1000

U0 = V_combined_lst_stochastic[-1]

try:
    stiefel_samples_file = base_path / f"{variable}_stiefel_samples_combined_KMeans.h5"
    if stiefel_samples_file.exists():
        print("Stiefel samples file found. Loading samples.")
        with h5py.File(stiefel_samples_file, "r") as stiefel_file:
            stiefel_samples_combined = stiefel_file["stiefel_samples"][:]
            maxI = stiefel_file["maxI"][:]
            beta = stiefel_file["beta"][:]
    
    else:
        print("Stiefel samples file not found. Generating new samples.")
        raise FileNotFoundError("Stiefel samples file not found.")

    print("Stiefel samples loaded successfully.")
    print("Stiefel samples shape:", stiefel_samples_combined.shape)
    print("MaxI shape:", maxI.shape)
    print("Beta paramters: ", beta)

except FileNotFoundError:
    print("Stiefel samples file not found. Generating new samples.")
    stiefel_samples_combined, maxI, beta = stiefel.gen_stiefel_samples(
        N_stiefel_samples,
        V_combined_lst_stochastic,
        tau=0.00001,
        metric_alpha=0.0000000000,
    )

    # Align signs of stiefel samples with global ROB
    for i in range(len(stiefel_samples_combined)):
        for j in range(stiefel_samples_combined[i].shape[1]):
            dist1 = np.linalg.norm(U0[:, j] - stiefel_samples_combined[i][:, j])
            dist2 = np.linalg.norm(U0[:, j] + stiefel_samples_combined[i][:, j])
            if dist2 < dist1:
                stiefel_samples_combined[i][:, j] = -stiefel_samples_combined[i][:, j]

    # stiefel_samples_Vr = np.array([sample[:, :r] for sample in stiefel_samples_combined])
    # stiefel_samples_Vbar = np.array(
    #     [sample[:, r : r + q_trunc] for sample in stiefel_samples_combined]
    # )

    # save the 1000 stiefel_samples_combined to a file
    stiefel_samples_file = base_path / f"{variable}_stiefel_samples_combined_KMeans.h5"
    with h5py.File(stiefel_samples_file, "w") as f:
        f.create_dataset("stiefel_samples", data=stiefel_samples_combined)
        f.create_dataset("maxI", data=maxI)
        f.create_dataset("beta", data=beta)

    print("Stiefel samples saved successfully.")

# %%
# append the frechet_mean to the dataset
try:
    stiefel_samples_file = base_path / f"{variable}_stiefel_samples_combined_KMeans.h5"
    if stiefel_samples_file.exists():
        print("stiefel_samples_file already exists")
        print("Loading Frechet mean from file")
        with h5py.File(stiefel_samples_file, "r") as f:
            frechet_mean = f["frechet_mean"][()]
            Vr_frechet_mean = f["Vr_frechet_mean"][()]
            Vbar_frechet_mean = f["Vbar_frechet_mean"][()]

        print("Frechet mean loaded successfully")
except:
    frechet_mean = calc_frechet_mean_mat(
        stiefel_samples_combined, V_combined_lst_stochastic[-1], eps=1e-2, tau=1e-3
    )
    Vr_frechet_mean = frechet_mean[0][:, :r]
    Vbar_frechet_mean = frechet_mean[0][:, r : +r + q_trunc]
    with h5py.File(stiefel_samples_file, "a") as f:
        f.create_dataset("frechet_mean", data=frechet_mean[0])
        f.create_dataset("Vr_frechet_mean", data=Vr_frechet_mean)
        f.create_dataset("Vbar_frechet_mean", data=Vbar_frechet_mean)

    print("Frechet mean calculated and saved")
