# %%
import numpy as np
import mpi4py
import sys
import scipy
import h5py
import tqdm
import adios4dolfinx
import dolfinx
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
import pickle as pkl

plt.style.use(["science", "grid"])
plt.rcParams.update({"font.size": 16})

# set numpy random seed
np.random.seed(42)
from Representation import *

import numpy as np


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



# -------------------------------
def calculate_statistics_hdf5(store_path, chunk_size=1000):
    with h5py.File(store_path, "r") as f:
        data = f["s_rec_full"]
        valid_mask = f["valid_mask"][:]
        total_samples = len(valid_mask)

        sum_s_rec = np.zeros(data.shape[1:])
        sum_squared = np.zeros(data.shape[1:])
        all_values = []
        count_valid = 0

        for i in tqdm.tqdm(range(0, total_samples, chunk_size), desc="Processing samples"):
            chunk_mask = valid_mask[i : min(i + chunk_size, total_samples)]
            chunk_data = data[i : min(i + chunk_size, total_samples)]

            valid_chunk = chunk_data[chunk_mask]

            if len(valid_chunk) > 0:
                sum_s_rec += np.sum(valid_chunk, axis=0)
                sum_squared += np.sum(valid_chunk**2, axis=0)
                all_values.extend(valid_chunk)
                count_valid += len(valid_chunk)

    if count_valid == 0:
        raise ValueError("No valid samples found!")

    mean_s_rec = sum_s_rec / count_valid
    variance = (sum_squared / count_valid) - (mean_s_rec**2)
    variance = np.maximum(variance, 0)
    std_s_rec = np.sqrt(variance)

    # Calculate 95% confidence interval using percentile method
    all_values = np.array(all_values)
    ci_lower = np.percentile(all_values, 2.5, axis=0)
    ci_upper = np.percentile(all_values, 97.5, axis=0)

    return mean_s_rec, std_s_rec, ci_lower, ci_upper, count_valid, total_samples


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


def calculate_velocity_magnitude_stats(store_path, chunk_size=1000):
    with h5py.File(store_path, "r") as f:
        data = f["s_rec_full"]
        valid_mask = f["valid_mask"][:]
        total_samples = len(valid_mask)

        sum_magnitude = 0
        sum_squared_magnitude = 0
        all_magnitudes = []
        count_valid = 0

        for i in tqdm.tqdm(range(0, total_samples, chunk_size), desc="Processing samples"):
            chunk_mask = valid_mask[i : min(i + chunk_size, total_samples)]
            chunk_data = data[i : min(i + chunk_size, total_samples)]

            valid_chunk = chunk_data[chunk_mask]

            if len(valid_chunk) > 0:
                for sample in valid_chunk:
                    magnitude = compute_magnitude(sample)
                    sum_magnitude += magnitude
                    sum_squared_magnitude += magnitude**2
                    all_magnitudes.append(magnitude)
                    count_valid += 1

    if count_valid == 0:
        raise ValueError("No valid samples found!")

    mean_magnitude = sum_magnitude / count_valid
    variance_magnitude = (sum_squared_magnitude / count_valid) - (mean_magnitude**2)
    variance_magnitude = np.maximum(variance_magnitude, 0)
    std_magnitude = np.sqrt(variance_magnitude)

    # Calculate 95% confidence interval using percentile method
    all_magnitudes = np.array(all_magnitudes)
    ci_lower = np.percentile(all_magnitudes, 2.5, axis=0)
    ci_upper = np.percentile(all_magnitudes, 97.5, axis=0)

    return mean_magnitude, std_magnitude, ci_lower, ci_upper, count_valid, total_samples


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


def compute_vorticity(u: Function, omega: Function, Q, state_all, timestep):
    u.x.array[:] = state_all[:, timestep]
    ux, uy = u.split()
    vortex = uy.dx(0) - ux.dx(1)
    omega.interpolate(Expression(vortex, Q.element.interpolation_points()))
    vorticity_values = omega.x.array[:].copy()

    return vorticity_values


# def compute_vorticity_all_timesteps(u: Function, omega: Function, Q, state_all):
#     vorticity = []
#     for timestep in range(state_all.shape[1]):
#         vorticity_values = compute_vorticity(u, omega, Q, state_all, timestep)
#         vorticity.append(vorticity_values)

#     # Transpose to have time steps as columns, and points as rows
#     vorticity = np.array(vorticity).T

#     return vorticity


def compute_vorticity_all_timesteps(mymesh, state_all):

    # compute vorticity
    v_cg2 = element(
        "Lagrange", mymesh.topology.cell_name(), 2, shape=(mymesh.geometry.dim,)
    )
    dg1 = element("DG", mymesh.topology.cell_name(), 1)

    V = functionspace(mymesh, v_cg2)
    Q = functionspace(mymesh, dg1)

    # Create functions for velocity and vorticity
    u = Function(V)
    omega = Function(Q)

    vorticity = []
    for timestep in range(state_all.shape[1]):
        vorticity_values = compute_vorticity(u, omega, Q, state_all, timestep)
        vorticity.append(vorticity_values)
        # print("Max vorticity: ", np.max(vorticity_values))

    # Transpose to have time steps as columns, and points as rows
    vorticity = np.array(vorticity).T

    return vorticity


import h5py
import numpy as np
import tqdm
from pathlib import Path

import h5py
import numpy as np
import tqdm
from pathlib import Path


def calculate_vorticity_stats(store_path, vorticity_path, mymesh, chunk_size=1000):
    vorticity_path = Path(vorticity_path)
    if vorticity_path.exists():
        compute_vort = False
    else:
        compute_vort = True

    with h5py.File(store_path, "r") as f:
        data = f["s_rec_full"]
        valid_mask = f["valid_mask"][:]
        total_samples = len(valid_mask)

        print(f"Shape of s_rec_full: {data.shape}")
        print(f"Shape of valid_mask: {valid_mask.shape}")
        print(f"Sum of valid_mask: {np.sum(valid_mask)}")
        print(f"First few elements of valid_mask: {valid_mask[:10]}")

        sum_vorticity = 0
        sum_squared_vorticity = 0
        count_valid = 0

        # Initialize updater

        # compute vorticity
        dg1 = element("DG", mymesh.topology.cell_name(), 1)
        Q = functionspace(mymesh, dg1)

        if compute_vort:
            print("Computing vorticity full")
            with h5py.File(vorticity_path, "w") as f_vort:
                f_vort.create_dataset(
                    "vorticity",
                    shape=(total_samples, len(Q.tabulate_dof_coordinates()), data.shape[2]),
                    dtype=np.float32,
                )
                f_vort.create_dataset("valid_mask", data=valid_mask, dtype=bool)

                for i in tqdm.tqdm(range(0, total_samples, chunk_size), desc="Processing samples"):
                    chunk_end = min(i + chunk_size, total_samples)
                    chunk_mask = valid_mask[i:chunk_end]
                    chunk_data = data[i:chunk_end]

                    for j, (is_valid, sample) in enumerate(zip(chunk_mask, chunk_data)):
                        if is_valid:
                            vorticity = compute_vorticity_all_timesteps(mymesh, sample)
                            f_vort["vorticity"][i + j] = vorticity

                            sum_vorticity += vorticity
                            sum_squared_vorticity += vorticity**2
                            count_valid += 1

                            # Update
                            

                    if i == 0 and np.any(chunk_mask):
                        print(f"Shape of first valid sample: {chunk_data[chunk_mask][0].shape}")

        else:
            with h5py.File(vorticity_path, "r") as f_vort:
                vorticity_data = f_vort["vorticity"]
                valid_mask = f_vort["valid_mask"][:]

                for i in tqdm.tqdm(range(0, total_samples, chunk_size), desc="Processing samples"):
                    chunk_end = min(i + chunk_size, total_samples)
                    chunk_mask = valid_mask[i:chunk_end]
                    chunk_vorticity = vorticity_data[i:chunk_end][chunk_mask]

                    sum_vorticity += np.sum(chunk_vorticity, axis=0)
                    sum_squared_vorticity += np.sum(chunk_vorticity**2, axis=0)
                    count_valid += np.sum(chunk_mask)

                    # Update
                    

    if count_valid == 0:
        raise ValueError("No valid samples found!")

    mean_vorticity = sum_vorticity / count_valid
    variance_vorticity = (sum_squared_vorticity / count_valid) - (mean_vorticity**2)

    variance_vorticity = np.maximum(variance_vorticity, 0)
    std_vorticity = np.sqrt(variance_vorticity)

    # Calculate 95% credibility interval



    return mean_vorticity, std_vorticity, lower_percentile, upper_percentile, count_valid, total_samples


# -------------------------------

recompute_state = True
recompute_mag = True
recompute_vort = True

# -------------------------------

# Load data
t_start = 4
T_end_train = 5

variable = "u"

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


Reynolds = [75, 112.5, 135, 150, 165, 187.5, 225, 262.5, 300]

mus_reynolds_dict = {mu: Re for mu, Re in zip(mus, Reynolds)}

test_mu = 0.000556
test_Re = int(np.round(1.5 * 0.1 * 1 / test_mu))
X_all_test = np.load(
    f"/data1/jy384/research/Data/UnimodalSROB/ns/Re_{test_Re}_mu_{test_mu}/{variable}_snapshots_matrix_test.npy"
).T

import yaml
from omegaconf import OmegaConf

base_path = Path("/data1/jy384/research/Data/UnimodalSROB/ns/")
effective_dts = {}

for mu in mus:
    Re = mus_reynolds_dict[mu]
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


mesh_path = base_path / f"Re_{test_Re}_mu_{test_mu}" / f"u_snapshots_mu{test_mu}.bp"
mymesh = read_mesh(mesh_path)


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
names = [f"$\mu$={mus}" for mus in drawn_mus]

fig, ax = plt.subplots(figsize=(8, 6))

err_tol = 5e-2


# %%
# Model parameters
r = 3  # min for 5e-2
q_trunc = 13
# p = 3

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


# Usage
store_path = base_path / f"{variable}_ReconStates.h5"
state_statistics_path = base_path / f"{variable}_s_rec_full_statistics.npz"
try:
    if state_statistics_path.exists():
        print(f"File for {variable} statistics exists!")
        if recompute_state:
            mean_s_rec, std_s_rec, lower_percentile, upper_percentile, valid_count, total_samples = (
                calculate_statistics_hdf5(
                    store_path, chunk_size=20  # Adjust chunk_size as needed
                )
            )
            print(
                f"Processed {valid_count} valid samples out of {total_samples} total samples."
            )

            # Save results
            np.savez_compressed(
                state_statistics_path,
                mean=mean_s_rec,
                std_dev=std_s_rec,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                valid_samples=valid_count,
                total_samples=total_samples,
            )

            print(f"Results saved to {state_statistics_path}")

    else:
        print(f"Cannot open {state_statistics_path}. File does not exist.")
        raise FileNotFoundError(
            f"FileNotFoundError, File for {variable} statistics does not exist!"
        )

except Exception as e:
    if "FileNotFoundError" in str(e):
        print(f"File for {variable} statistics does not exist.")
        mean_s_rec, std_s_rec, lower_percentile_s_rec, upper_percentile_s_rec, valid_count, total_samples = calculate_statistics_hdf5(
            store_path, chunk_size=20  # Adjust chunk_size as needed
        )
        print(
            f"Processed {valid_count} valid samples out of {total_samples} total samples."
        )

        # Save results
        np.savez_compressed(
            state_statistics_path,
            mean=mean_s_rec,
            std_dev=std_s_rec,
            lower_percentile=lower_percentile_s_rec,
            upper_percentile=upper_percentile_s_rec,
            valid_samples=valid_count,
            total_samples=total_samples,
        )

        print(f"Results saved to {state_statistics_path}")
    else:
        print(f"An error occurred: {e}")


magnitude_statistics_path = base_path / f"{variable}_velocity_magnitude_statistics.npz"
try:
    if magnitude_statistics_path.exists():
        print(f"File for magnitude statistics exists!")
        if recompute_mag:
            mean_magnitude, std_magnitude, lower_percentile_magnitude, upper_percentile_magnitude, valid_count, total_samples = (
                calculate_velocity_magnitude_stats(
                    store_path,
                    chunk_size=20,  # Adjust chunk_size as needed
                )
            )
            print(
                f"Processed {valid_count} valid samples out of {total_samples} total samples."
            )

            # Save results
            np.savez_compressed(
                magnitude_statistics_path,
                mean_magnitude=mean_magnitude,
                std_dev_magnitude=std_magnitude,
                lower_percentile=lower_percentile_magnitude,
                upper_percentile=upper_percentile_magnitude,
                valid_samples=valid_count,
                total_samples=total_samples,
            )

            print(f"Results saved to {magnitude_statistics_path}")

    else:
        raise FileNotFoundError(
            f"FileNotFoundError, File for magnitude statistics does not exist!"
        )


except Exception as e:
    print(f"Error: {e}")
    if "FileNotFoundError" in str(e):
        print("File for velocity magnitude statistics does not exist!")
        print("Calculating velocity magnitude statistics...")
        mean_magnitude, std_magnitude, lower_percentile_magnitude, upper_percentile_magnitude, valid_count, total_samples = (
            calculate_velocity_magnitude_stats(
                store_path, chunk_size=20  # Adjust chunk_size as needed
            )
        )
        print(
            f"Processed {valid_count} valid samples out of {total_samples} total samples."
        )

        # Save results
        np.savez_compressed(
            magnitude_statistics_path,
            mean_magnitude=mean_magnitude,
            std_dev_magnitude=std_magnitude,
            lower_percentile=lower_percentile_magnitude,
            upper_percentile=upper_percentile_magnitude,
            valid_samples=valid_count,
            total_samples=total_samples,
        )

        print(f"Results saved to {magnitude_statistics_path}")


# compute vorticity
v_cg2 = element(
    "Lagrange", mymesh.topology.cell_name(), 2, shape=(mymesh.geometry.dim,)
)
dg1 = element("DG", mymesh.topology.cell_name(), 1)

V = functionspace(mymesh, v_cg2)
Q = functionspace(mymesh, dg1)

# Create functions for velocity and vorticity
u = Function(V)
omega = Function(Q)


vorticity_statistics_path = base_path / f"{variable}_vorticity_statistics.npz"
vorticity_path = base_path / f"{variable}_ReconVorticity.h5"
try:
    if vorticity_statistics_path.exists():
        print(f"File for vorticity statistics exists!")
        if recompute_vort:
            mean_vorticity, std_vorticity, lower_percentile_vorticity, upper_percentile_vorticity, valid_count, total_samples = (
                calculate_vorticity_stats(
                    store_path,
                    vorticity_path,
                    # u=u,
                    # omega=omega,
                    # Q=Q,
                    mymesh=mymesh,
                    chunk_size=20,  # Adjust chunk_size as needed
                )
            )
            print(
                f"Processed {valid_count} valid samples out of {total_samples} total samples."
            )

            # Save results
            np.savez_compressed(
                vorticity_statistics_path,
                mean_vorticity=mean_vorticity,
                std_dev_vorticity=std_vorticity,
                lower_percentile=lower_percentile_vorticity,
                upper_percentile=upper_percentile_vorticity,
                valid_samples=valid_count,
                total_samples=total_samples,
            )

            print(f"Results saved to {vorticity_statistics_path}")

    else:
        raise FileNotFoundError(
            f"FileNotFoundError, File for vorticity statistics does not exist!"
        )

except Exception as e:
    print(f"Error: {e}")
    if "FileNotFoundError" in str(e):
        print("File for vorticity statistics does not exist!")
        print("Calculating vorticity statistics...")
        mean_vorticity, std_vorticity, lower_percentile_vorticity, upper_percentile_vorticity, valid_count, total_samples = (
            calculate_vorticity_stats(
                store_path,
                vorticity_path,
                # u,
                # omega,
                # Q,
                mymesh=mymesh,
                chunk_size=20,  # Adjust chunk_size as needed
            )
        )
        print(
            f"Processed {valid_count} valid samples out of {total_samples} total samples."
        )

        # Save results
        np.savez_compressed(
            vorticity_statistics_path,
            mean_vorticity=mean_vorticity,
            std_dev_vorticity=std_vorticity,
            lower_percentile=lower_percentile_vorticity,
            upper_percentile=upper_percentile_vorticity,
            valid_samples=valid_count,
            total_samples=total_samples,
        )

        print(f"Results saved to {vorticity_statistics_path}")
