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
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans

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

plt.style.use(["science", "grid", "no-latex"])
plt.rcParams.update({"font.size": 16})

# set numpy random seed
np.random.seed(3)


# %%
from Representation import *


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

variable = "p"

print("Variable: ", variable)

# Load the HDF5 file
h5_file_path = Path(
    f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_combined_dataset.h5"
)

with h5py.File(h5_file_path, "r") as f:
    mus = f.attrs["mu_values"]
    U_c = f.attrs["U_c"]
    L_c = f.attrs["L_c"]
    rho = f.attrs["rho"]

    # Get the shape of the data
    sample_group = f[f"mu_{mus[0]:.6f}"]
    data_shape = sample_group["snapshots"].shape

# Define parameters
t_start = 4
T_end_train = 5


# combined_dataset = np.load(
#     "/data1/jy384/research/Data/UnimodalSROB/ns/combined_dataset.npy"
# )

# X_all_test = np.load(
#     "/data1/jy384/research/Data/UnimodalSROB/ns/Re_180_mu_0.000556/u_snapshots_matrix_test.npy"
# ).T
test_mu = 0.000556
test_Re = int(np.round(1.0 * 0.1 * 1 / test_mu))
X_all_test = np.load(
    f"/data1/jy384/research/Data/UnimodalSROB/ns/Re_{test_Re}_mu_{test_mu}/{variable}_snapshots_matrix_test.npy"
).T

# %%
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
T_end = 5
Train_T = int(T_end_train / dt)

config = {
    "N": data_shape[0],
    "dt": 1e-3,
    "T_end": T_end,
    "mus": list(mus),
    "Mp": Mp,
    "K": T_end / dt,  # T_end / dt
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

# create color tags for the combinations
drawn_mus = [tup[0] for tup in flattened_tagged_combinations]
color_tags = []
for n_X in range(len(flattened_tagged_combinations)):
    mus_list = mus.tolist()
    mus_idx = [mus_list.index(mus_) for mus_ in flattened_tagged_combinations[n_X][0]]
    print(mus_idx)
    color_tags.append(flattened_tagged_combinations[n_X][1])


rel_err_SVD_lst = []
idx_lst = []
idx_lst_1e_3 = []
rob_lst = []
# names = [f"tap={taps}" for taps in numtaps] + ["Nominal"]
names = [f"$\mu$={mus}" for mus in drawn_mus]

err_tol = 5e-2

# for every combination of mus, identify index where error is less than tol
# Create an HDF5 file to store the SVD results
svd_results_file = Path(
    f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_svd_results.h5"
)

try:
    if svd_results_file.exists():
        with h5py.File(svd_results_file, "r") as f:
            if "U" in f:
                print("U matrix already exists in the file.")
            else:
                print("U matrix does not exist in the file.")
    else:
        print("svd_results.h5 does not exist.")
        raise FileNotFoundError

except Exception as e:
    print(f"An error occurred: {e}")
    print("Attempting to create the file...")

    # First loop: Compute and store SVD results
    with h5py.File(svd_results_file, "w") as f:
        for i, combo in tqdm.tqdm(enumerate(drawn_mus)):
            X = []

            with h5py.File(h5_file_path, "r") as data_f:
                for mu in combo:
                    grp = data_f[f"mu_{mu:.6f}"]
                    snapshot = grp["snapshots"][:]
                    X.append(snapshot)

            X = np.concatenate(X, axis=1)
            X_ref = np.mean(X, axis=1)[:, None]
            X_centered = X - X_ref

            U, S, Vt = torch.linalg.svd(
                torch.tensor(X_centered, device="cuda"),
                driver="gesvda",
                full_matrices=False,
            )

            # Store U, S, and Vt matrices in the HDF5 file
            grp = f.create_group(f"combo_{i}")
            grp.create_dataset("U", data=U.cpu().numpy(), compression="gzip")
            grp.create_dataset("S", data=S.cpu().numpy(), compression="gzip")
            grp.create_dataset("Vt", data=Vt.cpu().numpy(), compression="gzip")

            eigvals_SVD = S.cpu().numpy() ** 2 * (1 / (len(S) - 1))
            rel_err_SVD = 1 - (np.cumsum(eigvals_SVD) / np.sum(eigvals_SVD))
            rel_err_SVD_lst.append(rel_err_SVD)

            idx = np.where(rel_err_SVD < err_tol)[0][0] + 1
            idx_lst.append(idx)

            print("Min for 1e-3: ", np.where(rel_err_SVD < 1e-3)[0][0] + 1)
            idx_lst_1e_3.append(np.where(rel_err_SVD < 1e-3)[0][0] + 1)

        # Model parameters
        # r = 6  # min for 5e-2

    np.save(
        f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_idx_lst.npy", idx_lst
    )
    np.save(
        f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_idx_lst_1e_3.npy",
        idx_lst_1e_3,
    )

try:
    idx_lst = np.load(
        f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_idx_lst.npy"
    )
    idx_lst_1e_3 = np.load(
        f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_idx_lst_1e_3.npy"
    )
    print("Min idx = ", min(idx_lst))
    r = min(idx_lst)
    q_trunc = min(idx_lst_1e_3) - r

except FileNotFoundError:
    print(f"{variable}_idx_lst.npy or {variable}_idx_lst_1e_3.npy not found")

# np.save("/data1/jy384/research/Data/UnimodalSROB/ns/idx_lst.npy", idx_lst)
# np.save("/data1/jy384/research/Data/UnimodalSROB/ns/idx_lst_1e_3.npy", idx_lst_1e_3)
# p = 3

print(f"r = {r}, q_trunc = {q_trunc}, p = {p}")

tol = 1e-3  # tolerence for alternating minimization
gamma = 0.01  # regularization parameter\
max_iter = 100  # maximum number of iterations

Vr_lst = []
Vbar_lst = []
Shat_lst = []
Xi_lst = []
Poly_lst = []
energy_lst = []


def process_combo(i, combo, h5_file, svd_results_file, r, q_trunc, p, gamma):
    X = []
    for mu in combo:
        grp = h5_file[f"mu_{mu:.6f}"]
        # Use memory mapping to access the dataset
        snapshot = grp["snapshots"]
        # Create a NumPy memmap array that references the HDF5 dataset
        memmap_snapshot = np.array(snapshot, dtype=snapshot.dtype)
        X.append(memmap_snapshot)

    X = np.concatenate(X, axis=1)
    X_ref = np.mean(X, axis=1)[:, None]
    X_centered = X - X_ref

    with h5py.File(svd_results_file, "r") as f:
        U = f[f"combo_{i}/U"][:]

    Vr = U[:, :r]
    Vbar = U[:, r : r + q_trunc]

    q = Vr.T @ X_centered

    Proj_error = X_centered - (Vr @ q)
    Poly = np.concatenate(polynomial_form(q, p), axis=0)

    A = Poly @ Poly.T + gamma * np.identity((p - 1) * r)
    B = Vbar.T @ Proj_error @ Poly.T

    L = linalg.cholesky(A, lower=True)
    Xi = linalg.solve_triangular(L.T, linalg.solve_triangular(L, B.T, lower=True)).T

    energy = (
        np.linalg.norm(Vr @ q + (Vbar @ Xi @ Poly), "fro") ** 2
        / np.linalg.norm(X - X_ref, "fro") ** 2
    )

    Gamma_MPOD = X_ref + (Vr @ q) + (Vbar @ Xi @ Poly)
    reconstruction_error = relative_error(X, Gamma_MPOD, X_ref)

    return Vr, Vbar, q, Xi, Poly, energy, reconstruction_error


# Profile one pass of the loop
def profile_single_pass():
    i = 0  # Choose the first combo for profiling
    combo = drawn_mus[0]
    process_combo(i, combo, h5_file_path, svd_results_file, r, q_trunc, p, gamma)


# try searching for basis file, if exists, load it
try:
    basis_file = f"/data1/jy384/research/Data/UnimodalSROB/ns/{variable}_basis_r{r}_q{q_trunc}_p{p}_gamma{gamma}.h5"
    # test if the file exists
    with h5py.File(basis_file, "r") as f:
        # if total number of stored basis is the same as the total number of combinations
        if len(f.keys()) == len(drawn_mus):
            print("Basis file found")
        else:
            raise FileNotFoundError

except:
    print("Basis file not found")

    # then proceed to the loop
    # Open the basis file once, outside the loop
    with h5py.File(
        basis_file, "w"
    ) as basis_f:  # Use "w" mode for the first time to create the file
        with h5py.File(h5_file_path, "r") as h5_file:
            for i, combo in tqdm.tqdm(enumerate(drawn_mus)):
                result = process_combo(
                    i, combo, h5_file, svd_results_file, r, q_trunc, p, gamma
                )

                # Save the results
                grp = basis_f.create_group(f"combo_{i}")
                grp.create_dataset("Vr", data=result[0])
                grp.create_dataset("Vbar", data=result[1])
                grp.create_dataset("Shat", data=result[2])
                grp.create_dataset("Xi", data=result[3])
                grp.create_dataset("Poly", data=result[4])
                grp.create_dataset("energy", data=result[5])

                print(f"Saved combo {i} to basis file")

                # print memory usage of the arrays
                print(f"Vr: {sys.getsizeof(result[0])}")
                print(f"Vbar: {sys.getsizeof(result[1])}")
                print(f"q: {sys.getsizeof(result[2])}")
                print(f"Xi: {sys.getsizeof(result[3])}")
                print(f"Poly: {sys.getsizeof(result[4])}")
                print(f"energy: {sys.getsizeof(result[5])}")

    print(f"All combos saved to {basis_file}")


# if basis file exists, load it
with h5py.File(basis_file, "r") as basis_f:
    print("Loading basis file")
    for i in tqdm.tqdm(range(len(drawn_mus))):
        grp = basis_f[f"combo_{i}"]
        Vr_lst.append(grp["Vr"][:])
        Vbar_lst.append(grp["Vbar"][:])
        Xi_lst.append(grp["Xi"][:])


# # Run the profiler
# pr = cProfile.Profile()
# pr.enable()
# profile_single_pass()
# pr.disable()

# # Print the profiling results
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
# ps.print_stats(20)  # Print top 20 time-consuming functions
# print(s.getvalue())


# with h5py.File(svd_results_file, "r") as f:
#     for i, combo in tqdm.tqdm(enumerate(drawn_mus)):
#         X = []

#         with h5py.File(h5_file_path, "r") as data_f:
#             for mu in combo:
#                 grp = data_f[f"mu_{mu:.6f}"]
#                 snapshot = grp["snapshots"][:]
#                 X.append(snapshot)

#         X = np.concatenate(X, axis=1)
#         X_ref = np.mean(X, axis=1)[:, None]
#         X_centered = X - X_ref

#         # Load U matrix from HDF5 file
#         U = f[f"combo_{i}/U"][:]

#         Vr = U[:, :r]
#         Vbar = U[:, r : r + q_trunc]

#         q = Vr.T @ X_centered

#         Proj_error = X_centered - (Vr @ q)
#         Poly = np.concatenate(polynomial_form(q, p), axis=0)
#         # Xi = (
#         #     Vbar.T
#         #     @ Proj_error
#         #     @ Poly.T
#         #     @ np.linalg.inv(Poly @ Poly.T + gamma * np.identity((p - 1) * r))
#         # )

#         from scipy import linalg

#         # ... (previous code remains the same)

#         Proj_error = X_centered - (Vr @ q)
#         Poly = np.concatenate(polynomial_form(q, p), axis=0)

#         A = Poly @ Poly.T + gamma * np.identity((p - 1) * r)
#         B = Vbar.T @ Proj_error @ Poly.T

#         L = linalg.cholesky(A, lower=True)
#         Xi = linalg.solve_triangular(L.T, linalg.solve_triangular(L, B.T, lower=True)).T

#         # ... (rest of the code remains the same)

#         energy = (
#             np.linalg.norm(Vr @ q + (Vbar @ Xi @ Poly), "fro") ** 2
#             / np.linalg.norm(X - X_ref, "fro") ** 2
#         )

#         print(f"Snapshot energy: {energy:e}")

#         Gamma_MPOD = X_ref + (Vr @ q) + (Vbar @ Xi @ Poly)
#         print(f"\nReconstruction error: {relative_error(X, Gamma_MPOD, X_ref):.4%}")

#         Vr_lst.append(Vr)
#         Vbar_lst.append(Vbar)
#         Shat_lst.append(q)
#         Xi_lst.append(Xi)
#         Poly_lst.append(Poly)

#         # q, energy, Xi, Vr, Vbar, Poly = alternating_minimization(X, X_ref, num_snapshots, max_iter, 1e-3, gamma, r, q, p, initial_Shat=None)

#         # print("q = ", q.shape)
#         # print("qbar = ", qbar.shape)


from scipy import sparse

ZERO_THRESHOLD = 1e-6


def analyze_sparsity(matrix):
    total_elements = matrix.size
    near_zero_elements = np.sum(np.abs(matrix) < ZERO_THRESHOLD)
    sparsity = near_zero_elements / total_elements
    return sparsity, near_zero_elements, total_elements


def check_sparsity(basis_file):
    with h5py.File(basis_file, "r") as f:
        n_combos = len(f.keys())

        for i in range(n_combos):
            print(f"Analyzing combo {i}:")

            # Analyze Vr
            Vr = f[f"combo_{i}/Vr"][:]
            sparsity_Vr, near_zero_Vr, total_Vr = analyze_sparsity(Vr)
            print(
                f"  Vr: Sparsity = {sparsity_Vr:.2%}, Near-zero elements: {near_zero_Vr}/{total_Vr}"
            )

            # Analyze Vbar
            Vbar = f[f"combo_{i}/Vbar"][:]
            sparsity_Vbar, near_zero_Vbar, total_Vbar = analyze_sparsity(Vbar)
            print(
                f"  Vbar: Sparsity = {sparsity_Vbar:.2%}, Near-zero elements: {near_zero_Vbar}/{total_Vbar}"
            )

            # Optionally, analyze combined V
            V_combined = np.concatenate([Vr, Vbar], axis=1)
            sparsity_V, near_zero_V, total_V = analyze_sparsity(V_combined)
            print(
                f"  V_combined: Sparsity = {sparsity_V:.2%}, Near-zero elements: {near_zero_V}/{total_V}"
            )

            print()  # Empty line for readability

            isSparse = sparsity_Vr < 0.99 and sparsity_Vbar < 0.99 and sparsity_V < 0.99

            return isSparse


def print_hdf5_structure(file):
    def print_group(name, obj):
        print(name)

    with h5py.File(file, "r") as f:
        f.visititems(print_group)


def convert_to_sparse(matrix):
    return sparse.csr_matrix(matrix)


def store_sparse_matrix(file, name, matrix):
    grp = file.create_group(name)
    grp.create_dataset("data", data=matrix.data)
    grp.create_dataset("indices", data=matrix.indices)
    grp.create_dataset("indptr", data=matrix.indptr)
    grp.attrs["shape"] = matrix.shape


def load_sparse_matrix(file, name):
    grp = file[name]
    return sparse.csr_matrix(
        (grp["data"][:], grp["indices"][:], grp["indptr"][:]), shape=grp.attrs["shape"]
    )


# print_hdf5_structure(basis_file)

# # Usage
# isSparse = check_sparsity(basis_file=basis_file)
# if isSparse:
#     # Example usage
#     with h5py.File(basis_file, "r") as f:
#         for i in range(len(drawn_mus)):
#             Vr = f[f"combo_{i}/Vr"][:]
#             Vbar = f[f"combo_{i}/Vbar"][:]

#             Vr_sparse = convert_to_sparse(Vr)
#             print(f"Dense shape: {Vr.shape}, Sparse shape: {Vr_sparse.shape}")
#             print(
#                 f"Memory usage: Dense: {Vr.nbytes / 1e6:.2f} MB, Sparse: {Vr_sparse.data.nbytes / 1e6:.2f} MB"
#             )

# # Usage for storing
# sparse_basis_file = f"/data1/jy384/research/Data/UnimodalSROB/ns/sparse_basis_r{r}_q{q_trunc}_p{p}_gamma{gamma}.h5"
# with h5py.File(sparse_basis_file, "w") as f:
#     for i in range(len(drawn_mus)):
#         Vr = f[f"combo_{i}/Vr"][:]
#         Vbar = f[f"combo_{i}/Vbar"][:]
#         Vr_sparse = convert_to_sparse(Vr)
#         Vbar_sparse = convert_to_sparse(Vbar)
#         store_sparse_matrix(f, f"combo_{i}/Vr", Vr_sparse)
#         store_sparse_matrix(f, f"combo_{i}/Vbar", Vbar_sparse)


if Vbar_lst is not None:

    Vr1 = Vr_lst[-1]
    for idx in range(len(Vr_lst) - 1):
        Vr_idx = Vr_lst[idx]
        for j in range(Vr_idx.shape[1]):
            dist1 = np.linalg.norm(Vr1[:, j] - Vr_idx[:, j])
            dist2 = np.linalg.norm(Vr1[:, j] + Vr_idx[:, j])
            if dist2 < dist1:
                Vr_lst[idx][:, j] = -Vr_lst[idx][:, j]

    Vbar1 = Vbar_lst[-1]
    for idx in range(len(Vbar_lst) - 1):
        Vbar_idx = Vbar_lst[idx]
        for j in range(Vbar_idx.shape[1]):
            dist1 = np.linalg.norm(Vbar1[:, j] - Vbar_idx[:, j])
            dist2 = np.linalg.norm(Vbar1[:, j] + Vbar_idx[:, j])
            if dist2 < dist1:
                Vbar_lst[idx][:, j] = -Vbar_lst[idx][:, j]


# else:
#     Vbar_lst = []
#     # Aligning the signs of Vr_lst with Vr_lst[-1]
#     with h5py.File(basis_file, "r") as f:
#         # Vr1 = Vr_lst[-1]
#         last_combo_idx = len(drawn_mus) - 1
#         Vr1 = f[f"combo_{last_combo_idx}/Vr"][:]

#         for idx in range(len(drawn_mus)):
#             Vr_idx = f[f"combo_{idx}/Vr"][:]
#             for j in range(Vr_idx.shape[1]):
#                 dist1 = np.linalg.norm(Vr1[:, j] - Vr_idx[:, j])
#                 dist2 = np.linalg.norm(Vr1[:, j] + Vr_idx[:, j])
#                 if dist2 < dist1:
#                     f[f"combo_{idx}/Vr"][:, j] = -f[f"combo_{idx}/Vr"][:, j]

#         # For Vbar_lst, ensuring sign agreement with Vbar_lst[-1]
#         Vbar1 = f[f"combo_{last_combo_idx}/Vbar"][:]
#         for idx in range(len(drawn_mus)):
#             Vbar_idx = f[f"combo_{idx}/Vbar"][:]
#             for j in range(Vbar_idx.shape[1]):
#                 dist1 = np.linalg.norm(Vbar1[:, j] - Vbar_idx[:, j])
#                 dist2 = np.linalg.norm(Vbar1[:, j] + Vbar_idx[:, j])
#                 if dist2 < dist1:
#                     f[f"combo_{idx}/Vbar"][:, j] = -f[f"combo_{idx}/Vbar"][:, j]


# for idx in range(len(Vr_lst) - 1):
#     Vr_idx = Vr_lst[idx]
#     for j in range(Vr_idx.shape[1]):
#         dist1 = np.linalg.norm(Vr1[:, j] - Vr_idx[:, j])
#         dist2 = np.linalg.norm(Vr1[:, j] + Vr_idx[:, j])
#         if dist2 < dist1:
#             Vr_lst[idx][:, j] = -Vr_lst[idx][:, j]
# %%
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

# %%
import importlib
import OpInf

# import reloading
# reload the whole OpInf module
importlib.reload(OpInf)
from OpInf import *


config["robparams"] = {"r": int(r)}
recompute_operators = False

# try loading the basis and operators from the h5 file
try:
    basis_operators_file = (
        base_path / f"{variable}_basis_operators_mu_{test_mu:.6f}.hdf5"
    )
    print("Checking if file at", basis_operators_file, "exists")
    if basis_operators_file.exists():
        if not recompute_operators:
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

        else:
            print("Recomputing operators...")
            operators_lst = []

            # use each mu in between the min and max mu
            with h5py.File(h5_file_path, "r") as file:
                for i, combo in tqdm.tqdm(enumerate(drawn_mus)):
                    # i = i_ + 11
                    X = []
                    for mu in combo:
                        grp = file[f"mu_{mu:.6f}"]
                        snapshot = grp["snapshots"][:]
                        X.append(snapshot)  # Transpose to match your original format

                    X = np.concatenate(X, axis=1)
                    X_ref = np.mean(X, axis=1)[:, None]
                    X_centered = X - X_ref

                    # load basis from basis_file h5 file
                    with h5py.File(basis_file, "r") as f:
                        Vr = f[f"combo_{i}/Vr"][:]
                        VVbar = f[f"combo_{i}/Vbar"][:]

                    q = Vr.T @ X_centered

                    Mp = len(drawn_mus[i])
                    print("Mp: ", Mp)

                    Nsnapshots = X.shape[1]
                    print("Nsnapshots: ", Nsnapshots)

                    dShatdt = []
                    Shat_lst = []
                    dSdt = []
                    for j in range(Mp):
                        start_ind = int((j) * Nsnapshots / Mp)
                        end_ind = int((j + 1) * Nsnapshots / Mp)
                        print("start_ind: ", start_ind)
                        print("end_ind: ", end_ind)
                        ddtshat, ind = ddt(q[:, start_ind:end_ind], dt=dt, scheme="4c")
                        ddts, ind = ddt(X[:, start_ind:end_ind], dt=dt, scheme="4c")
                        dShatdt.append(ddtshat)
                        ind = np.array(ind) + int((j) * Nsnapshots / Mp)
                        Shat_lst.append(q[:, ind])
                        dSdt.append(ddts)

                    # get Shat_true
                    q_test = Vr.T @ (X_all_test - X_ref)
                    ddtshat_test, ind = ddt(q_test, dt=dt, scheme="4c")
                    Shat_true = q_test[:, ind]

                    # update config file with truncation order r
                    config["robparams"] = {"r": int(r)}

                    Shat_py = np.concatenate(Shat_lst, axis=1)
                    dShatdt_py = np.concatenate(dShatdt, axis=1).T
                    dSdt_py = np.hstack(dSdt)

                    print("Shape of Shat_py: ", Shat_py.shape)
                    print("Shape of dShatdt_py: ", dShatdt_py.shape)

                    N = int(config["N"])
                    dt = effective_dts[mus[0]]
                    K = int(config["K"])
                    DS = config["DS"]
                    params = config["params"]  # This will be a dictionary in Python
                    robparams = config[
                        "robparams"
                    ]  # This will be a dictionary in Python

                    q0 = Vr.T @ (X_all_test[:, 0, None] - X_ref).flatten()
                    time_domain = np.arange(t_start, T_end_train, dt)
                    train_size = Shat_py.shape[1] // len(mus)

                    print("Train size: ", train_size)
                    regs_product = [
                        1e-1,
                        1e-1,
                        1,
                        1e5,
                        1e8,
                        10,
                        1e11,
                        1e15,
                        3,
                    ]

                    regs, errors = train_gridsearch(
                        Shat_py,
                        dShatdt_py,
                        Shat_true,
                        train_size,
                        r,
                        regs_product,
                        time_domain,
                        q0,
                        params,
                        testsize=None,
                        margin=1.1,
                    )

                    print(f"Regularization params: {regs}, \t Error: {errors}")

                    params["lambda1"] = regs[0]
                    params["lambda2"] = regs[1]
                    if len(regs) > 2:
                        params["lambda3"] = regs[2]

                    operators = infer_operators_nl(
                        Shat_py, None, config["params"], dShatdt_py
                    )

                    operators_lst.append(operators)

            print("Finished computing all operators...")

            # Save the operators and basis in a HDF5 file
            # Define the path for the new HDF5 file
            basis_operators_file = (
                base_path / f"{variable}_basis_operators_mu_{test_mu:.6f}.hdf5"
            )

            # Save Vr_lst, Vbar_lst, and operators_lst
            with h5py.File(basis_operators_file, "w") as f:
                f.create_dataset("Vr_lst", data=np.array(Vr_lst))
                f.create_dataset("Vbar_lst", data=np.array(Vbar_lst))

                operators_grp = f.create_group("operators_lst")
                for i, ops in enumerate(operators_lst):
                    ops_grp = operators_grp.create_group(f"combo_{i}")
                    for key, value in ops.items():
                        ops_grp.create_dataset(key, data=value)

            print("Saved basis and operators data")

    else:
        raise Exception("Basis and operators data not found")

except:
    print("Could not load basis and operators data")

    operators_lst = []

    # use each mu in between the min and max mu
    with h5py.File(h5_file_path, "r") as file:
        for i, combo in tqdm.tqdm(enumerate(drawn_mus)):
            # i = i_ + 11
            X = []
            for mu in combo:
                grp = file[f"mu_{mu:.6f}"]
                snapshot = grp["snapshots"][:]
                X.append(snapshot)  # Transpose to match your original format

            X = np.concatenate(X, axis=1)
            X_ref = np.mean(X, axis=1)[:, None]
            X_centered = X - X_ref

            # load basis from basis_file h5 file
            with h5py.File(basis_file, "r") as f:
                Vr = f[f"combo_{i}/Vr"][:]
                VVbar = f[f"combo_{i}/Vbar"][:]

            q = Vr.T @ X_centered

            Mp = len(drawn_mus[i])
            print("Mp: ", Mp)

            Nsnapshots = X.shape[1]
            print("Nsnapshots: ", Nsnapshots)

            dShatdt = []
            Shat_lst = []
            dSdt = []
            for j in range(Mp):
                start_ind = int((j) * Nsnapshots / Mp)
                end_ind = int((j + 1) * Nsnapshots / Mp)
                print("start_ind: ", start_ind)
                print("end_ind: ", end_ind)
                ddtshat, ind = ddt(q[:, start_ind:end_ind], dt=dt, scheme="4c")
                ddts, ind = ddt(X[:, start_ind:end_ind], dt=dt, scheme="4c")
                dShatdt.append(ddtshat)
                ind = np.array(ind) + int((j) * Nsnapshots / Mp)
                Shat_lst.append(q[:, ind])
                dSdt.append(ddts)

            # get Shat_true
            q_test = Vr.T @ (X_all_test - X_ref)
            ddtshat_test, ind = ddt(q_test, dt=dt, scheme="4c")
            Shat_true = q_test[:, ind]

            # update config file with truncation order r
            config["robparams"] = {"r": int(r)}

            Shat_py = np.concatenate(Shat_lst, axis=1)
            dShatdt_py = np.concatenate(dShatdt, axis=1).T
            dSdt_py = np.hstack(dSdt)

            print("Shape of Shat_py: ", Shat_py.shape)
            print("Shape of dShatdt_py: ", dShatdt_py.shape)

            N = int(config["N"])
            dt = effective_dts[mus[0]]
            K = int(config["K"])
            DS = config["DS"]
            params = config["params"]  # This will be a dictionary in Python
            robparams = config["robparams"]  # This will be a dictionary in Python

            q0 = Vr.T @ (X_all_test[:, 0, None] - X_ref).flatten()
            time_domain = np.arange(t_start, T_end_train, dt)
            train_size = Shat_py.shape[1] // len(mus)

            print("Train size: ", train_size)
            regs_product = [
                1e-1,
                1e-1,
                1,
                1e5,
                1e8,
                10,
                1e11,
                1e15,
                3,
            ]

            regs, errors = train_gridsearch(
                Shat_py,
                dShatdt_py,
                Shat_true,
                train_size,
                r,
                regs_product,
                time_domain,
                q0,
                params,
                testsize=None,
                margin=1.1,
            )

            print(f"Regularization params: {regs}, \t Error: {errors}")

            params["lambda1"] = regs[0]
            params["lambda2"] = regs[1]
            if len(regs) > 2:
                params["lambda3"] = regs[2]

            operators = infer_operators_nl(Shat_py, None, config["params"], dShatdt_py)

            operators_lst.append(operators)

    print("Finished computing all operators...")

    # Save the operators and basis in a HDF5 file
    # Define the path for the new HDF5 file
    basis_operators_file = (
        base_path / f"{variable}_basis_operators_mu_{test_mu:.6f}.hdf5"
    )

    # Save Vr_lst, Vbar_lst, and operators_lst
    with h5py.File(basis_operators_file, "w") as f:
        f.create_dataset("Vr_lst", data=np.array(Vr_lst))
        f.create_dataset("Vbar_lst", data=np.array(Vbar_lst))

        operators_grp = f.create_group("operators_lst")
        for i, ops in enumerate(operators_lst):
            ops_grp = operators_grp.create_group(f"combo_{i}")
            for key, value in ops.items():
                ops_grp.create_dataset(key, data=value)

    print("Saved basis and operators data")


T_end_index = int((T_end_train - t_start) / dt)

# # Analyze difference in operators
# for each operator compute pairwise difference in terms of Frobenius Norm, Spectral Norm, and Eigenvalue distances
operatorsA = []
operatorsF = []
operatorsC = []
operatorsP = []

# Extract matrices from the dictionaries
for dictionary in operators_lst:
    operatorsA.append(dictionary["A"])
    operatorsF.append(dictionary["F"])
    operatorsC.append(dictionary["C"])
    operatorsP.append(dictionary["P"])


# Function to compute the normalized Frobenius norm difference between two matrices
def frobenius_norm_difference(matrix_a, matrix_b):
    norm_a = np.linalg.norm(matrix_a, "fro")
    norm_b = np.linalg.norm(matrix_b, "fro")
    diff_norm = np.linalg.norm(matrix_a - matrix_b, "fro")
    return diff_norm


# Function to compute the normalized Spectral norm difference between two matrices
def spectral_norm_difference(matrix_a, matrix_b):
    norm_a = np.linalg.norm(matrix_a)
    norm_b = np.linalg.norm(matrix_b)
    diff_norm = np.linalg.norm(matrix_a - matrix_b)
    return diff_norm


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


# Calculate pairwise distances for each type of operator and each norm
operatorsA_frobenius = pairwise_mat_distances(operatorsA, frobenius_norm_difference)
operatorsF_frobenius = pairwise_mat_distances(operatorsF, frobenius_norm_difference)
operatorsC_frobenius = pairwise_mat_distances(operatorsC, frobenius_norm_difference)
operatorsP_frobenius = pairwise_mat_distances(operatorsP, frobenius_norm_difference)

operatorsA_spectral = pairwise_mat_distances(operatorsA, spectral_norm_difference)
operatorsF_spectral = pairwise_mat_distances(operatorsF, spectral_norm_difference)
operatorsC_spectral = pairwise_mat_distances(operatorsC, spectral_norm_difference)
operatorsP_spectral = pairwise_mat_distances(operatorsP, spectral_norm_difference)

# Print or process the resulting distance matrices as needed
print("Pairwise Frobenius distances (A):")
print(operatorsA_frobenius)

print("Pairwise Spectral distances (A):")
print(operatorsA_spectral)

# Repeat for F, C, and P as needed
# calculate total frobenius norm of operators
operators_total_frobenius = (
    operatorsA_frobenius
    + operatorsF_frobenius
    + operatorsC_frobenius
    + operatorsP_frobenius
)

# Create a heatmap to visualize the distance matrix
num_matrices = len(operatorsA)
# %%
# for each combination, sum the pairwise distances and plot the spectral embedding colored by the sum of pairwise distances
pairwise_sum = operators_total_frobenius.sum(axis=1)
V_combined_embedding = embedder.fit_transform(flat_V_combined_arr.T)


# %%
# calculate the normalized frobenius norm difference between the V_combined basis
V_combined_frobenius = pairwise_mat_distances(V_combined_lst, frobenius_norm_difference)

# plot the spectral embedding of the V_combined basis colored by the normalized frobenius norm difference
V_combined_embedding = embedder.fit_transform(flat_V_combined_arr.T)


# %%
# cluster the V_combined embedding
# Clustering
n_clusters = 3  # Number of clusters to ensure distinct outliers
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(V_combined_embedding)

# for each cluster, find the operator with the least total frobenius norm
representatives = []
for i in range(n_clusters):
    cluster_indices = np.where(cluster_labels == i)[0]
    # check if all the selected indices are indeed in the cluster
    # cluster_distances = V_combined_frobenius[np.ix_(cluster_indices, cluster_indices)]
    cluster_distances = operators_total_frobenius[
        np.ix_(cluster_indices, cluster_indices)
    ]
    representatives.append(
        cluster_indices[np.argsort(cluster_distances.sum(axis=1))[0]]
    )

# Check that all representatives belong to the correct cluster
assert np.all(cluster_labels[representatives] == np.arange(n_clusters))

print(f"The representatives for each cluster are at indices: {representatives}")


# ---------------
# # Clustering
# n_clusters = 3  # Number of clusters to ensure distinct outliers
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# cluster_labels = kmeans.fit_predict(V_combined_embedding)

# # Function to calculate inter-cluster distance
# def inter_cluster_distance(point, other_cluster_points):
#     return np.mean([np.linalg.norm(point - p) for p in other_cluster_points])

# # Function to calculate intra-cluster cohesion
# def intra_cluster_cohesion(point, cluster_points):
#     return np.mean([np.linalg.norm(point - p) for p in cluster_points])

# representatives = []
# for i in range(n_clusters):
#     cluster_indices = np.where(cluster_labels == i)[0]
#     cluster_points = V_combined_embedding[cluster_indices]
#     other_cluster_points = V_combined_embedding[cluster_labels != i]

#     # Calculate scores for each point in the cluster
#     scores = []
#     for idx, point in zip(cluster_indices, cluster_points):
#         inter_distance = inter_cluster_distance(point, other_cluster_points)
#         intra_cohesion = intra_cluster_cohesion(point, cluster_points)
#         operator_robustness = 1 / (operators_total_frobenius[idx].sum() + 1e-6)  # Add small epsilon to avoid division by zero

#         # Combine scores (you can adjust the weights)
#         score = inter_distance - 0.5 * intra_cohesion + 0.5 * operator_robustness
#         scores.append((idx, score))

#     # Select the point with the highest score as the representative
#     representatives.append(max(scores, key=lambda x: x[1])[0])

# # Check that all representatives belong to the correct cluster
# assert np.all(cluster_labels[representatives] == np.arange(n_clusters))

# print(f"The representatives for each cluster are at indices: {representatives}")
# ---------------

# # %%
# # riemannian k-means with geomstats
# from geomstats.learning.kmeans import RiemannianKMeans
# from geomstats.geometry.stiefel import Stiefel
# from Stiefel_Exp_Log import distStiefel

# # Create a Stiefel manifold
# np.random.seed(3)
# stiefel_man = Stiefel(V_combined_lst[0].shape[0], V_combined_lst[0].shape[1])
# # while the minimum of the *number of points in each cluster* is <= 5, keep doing RiemKMeans and increase number of cluster.
# riemKMeansStopCond = True
# silhouette_scores = []
# min_n_clusters = 3
# n_clusters = min_n_clusters
# representatives = []

# V_combined_arr = np.array(V_combined_lst)

# while riemKMeansStopCond:
#     frechet_vars = []
#     karcher_means = []
#     riem_kmeans = RiemannianKMeans(stiefel_man, n_clusters, tol=1e-2, verbose=1)
#     riem_kmeans.fit(V_combined_arr)
#     cluster_labels = riem_kmeans.predict(V_combined_arr)
#     cluster_centers = riem_kmeans.centroids_

#     # calculate silhouette score
#     # frechet_variace_Ch
#     for i in range(n_clusters):
#         cluster_indices = np.where(cluster_labels == i)[0]

#         cluster_mean = calc_frechet_mean_mat(
#             V_combined_arr[cluster_indices], cluster_centers[i], eps=1e-3
#         )[0]
#         geodesic_distances = np.array(
#             [distStiefel(cluster_mean, V_combined_arr[j]) for j in cluster_indices]
#         )
#         frechet_var = np.mean(geodesic_distances**2)
#         frechet_vars.append(frechet_var)
#         karcher_means.append(cluster_mean)

#         # check if all the selected indices are indeed in the cluster
#         # cluster_distances = V_combined_frobenius[np.ix_(cluster_indices, cluster_indices)]
#         cluster_distances = operators_total_frobenius[
#             np.ix_(cluster_indices, cluster_indices)
#         ]
#         representatives.append(
#             cluster_indices[np.argsort(cluster_distances.sum(axis=1))[0]]
#         )

#     karcher_means = np.array(karcher_means)
#     frechet_var_of_Karcher_mean = np.mean(karcher_means**2)
#     score = frechet_var_of_Karcher_mean / np.sum(
#         frechet_vars[min_n_clusters - 1 : n_clusters]
#     )
#     silhouette_scores.append(score)

#     riemKMeansStopCond = (
#         min([len(np.where(cluster_labels == i)[0]) for i in range(n_clusters)]) <= 5
#     )
#     n_clusters += 1

# optimal_n_clusters = np.argmax(silhouette_scores) + min_n_clusters

# riem_kmeans = RiemannianKMeans(stiefel_man, optimal_n_clusters, tol=1e-2, verbose=1)
# riem_kmeans.fit(np.array(V_combined_lst))


# # %%
# cluster_labels = riem_kmeans.labels_

# # for each cluster, find the operator with the least total frobenius norm
# representatives = []
# for i in range(optimal_n_clusters):
#     cluster_indices = np.where(cluster_labels == i)[0]
#     # check if all the selected indices are indeed in the cluster
#     # cluster_distances = V_combined_frobenius[np.ix_(cluster_indices, cluster_indices)]
#     cluster_distances = operators_total_frobenius[
#         np.ix_(cluster_indices, cluster_indices)
#     ]
#     representatives.append(
#         cluster_indices[np.argsort(cluster_distances.sum(axis=1))[0]]
#     )

# # Check that all representatives belong to the correct cluster
# assert np.all(cluster_labels[representatives] == np.arange(optimal_n_clusters))

# print(f"The representatives for each cluster are at indices: {representatives}")

representatives_data = {
    "indices": representatives,
    "cluster_labels": cluster_labels,
    "method": "RiemannianKMeans",
    "n_clusters": n_clusters,
}

with open(base_path / f"{variable}_representatives_data_KMeans.pkl", "wb") as f:
    pkl.dump(representatives_data, f)

print("Saved representatives with metadata")

# # get distinct operators and evaluate
abs_error_full_lst_operators = []
relative_error_testing_window_lst_operators = []
relative_error_training_window_lst_operators = []
s_rec_full_lst_operators = []


# Define the path for the new HDF5 file to store s_rec_full
s_rec_full_file = base_path / f"{variable}_s_rec_full_mu_{test_mu:.6f}_KMeans.h5"

import psutil


def calculate_optimal_chunk_size(data_shape, dtype, available_memory_fraction=0.5):
    """
    Calculate an optimal chunk size based on data shape and available system memory.

    :param data_shape: Shape of the full dataset
    :param dtype: Data type of the dataset
    :param available_memory_fraction: Fraction of available memory to use (default 0.5)
    :return: Tuple of chunk sizes for each dimension
    """
    # Get available system memory
    available_memory = psutil.virtual_memory().available

    # Calculate size of one data point
    element_size = np.dtype(dtype).itemsize

    # Calculate total size of the dataset
    total_size = np.prod(data_shape) * element_size

    # Calculate target chunk size (in bytes)
    target_chunk_size = available_memory * available_memory_fraction

    # Ensure the chunk size is not larger than the dataset
    target_chunk_size = min(target_chunk_size, total_size)

    # Calculate the number of chunks
    num_chunks = max(1, int(np.ceil(total_size / target_chunk_size)))

    # Calculate chunk shape
    chunk_shape = tuple(
        max(1, int(dim / num_chunks ** (1 / len(data_shape)))) for dim in data_shape
    )

    return chunk_shape


# Open the file in write mode
if not s_rec_full_file.exists():
    with h5py.File(s_rec_full_file, "w") as s_rec_file:
        with h5py.File(basis_file, "r") as basis_file_handle:
            with h5py.File(h5_file_path, "r") as h5_file:
                for i, combo in tqdm.tqdm(enumerate(drawn_mus)):
                    X_ref = np.zeros(
                        (h5_file[f"mu_{combo[0]:.6f}/snapshots"].shape[0], 1)
                    )
                    total_snapshots = 0

                    # Compute X_ref
                    for mu in combo:
                        grp = h5_file[f"mu_{mu:.6f}"]
                        X_ref += np.sum(grp["snapshots"][:], axis=1, keepdims=True)
                        total_snapshots += grp["snapshots"].shape[1]
                    X_ref /= total_snapshots

                    Vr = Vr_lst[i]
                    Vbar = Vbar_lst[i]

                    # basis_grp = basis_file_handle[f"combo_{i}"]
                    # Xi = basis_grp["Xi"][:]
                    Xi = Xi_lst[i]

                    # print("Size of Xi:  ", sys.getsizeof(Xi) / 1e9)

                    operators = operators_lst[i]

                    q0 = Vr.T @ (X_all_test[:, 0, None] - X_ref).flatten()

                    T_end_full = 8
                    time_domain_full = np.arange(t_start, T_end_full, dt)

                    multi_indices = generate_multi_indices_efficient(len(q0), p=p)

                    out_full = scipy.integrate.solve_ivp(
                        rhs,
                        [time_domain_full[0], time_domain_full[-1]],
                        q0,
                        t_eval=time_domain_full,
                        args=[operators, config["params"], None, multi_indices],
                    )

                    s_hat_full = out_full.y
                    del out_full

                    # Compute s_rec_full in chunks to save memory
                    chunk_size = 1000  # Adjust based on your memory constraints
                    num_chunks = (s_hat_full.shape[1] + chunk_size - 1) // chunk_size

                    # Create a dataset for this combo
                    s_rec_dataset = s_rec_file.create_dataset(
                        f"combo_{i}",
                        shape=(Vr.shape[0], s_hat_full.shape[1]),
                        dtype=np.float32,
                        chunks=True,
                    )

                    poly_full = np.concatenate(polynomial_form(s_hat_full, p=p), axis=0)
                    s_rec_full = X_ref + Vr @ s_hat_full + Vbar @ Xi @ poly_full
                    s_rec_dataset[:] = s_rec_full

                    print(f"Saved s_rec_full for combo {i}")

    nNone = 0
    for arr in s_rec_full_lst_operators:
        if arr is None:
            nNone += 1
    print("Number of Nones:", nNone)
    print("Finished processing all combos")

else:
    print("s_rec_full file already exists. Skipping...")
