# %%
from OpInf import *
import numpy as np
import scipy
import h5py
import umap
import adios4dolfinx
import dolfinx
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
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

# Load data
t_start = 4
T_end_train = 5
dt = 0.000625
Train_T = int(T_end_train / dt)


combined_dataset = np.load(
    "/data1/jy384/research/Data/UnimodalSROB/ns/combined_dataset.npy"
)

X_all_test = np.load(
    "/data1/jy384/research/Data/UnimodalSROB/ns/Re_180_mu_0.000556/u_snapshots_matrix_test.npy"
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
    metadata_path = base_path / folder_name / "snapshots_matrix_train_metadata.yaml"

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
T_end = 2

config = {
    "N": combined_dataset[0].shape[0],
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

# %%
np.concatenate([X for X in combined_dataset], axis=1).shape

# %%
X_combined = np.concatenate([X for X in combined_dataset], axis=0).T

# randomly draw 3 ICs (mus) without replacement from X_all_nominal
# X_all_nominals_lst = [X_all_nominal, X_all_nominal_2]
X_list = []
drawn_mus = [tup[0] for tup in flattened_tagged_combinations]
color_tags = []
for n_X in range(len(flattened_tagged_combinations)):
    mus_idx = [mus.index(mus_) for mus_ in flattened_tagged_combinations[n_X][0]]
    print(mus_idx)
    color_tags.append(flattened_tagged_combinations[n_X][1])
    X_list.append(np.concatenate([combined_dataset[i].T for i in mus_idx], axis=1))

# X_list.append(X_nominal)
# color_tags.append((0,1,1)) # cyan

rob_lst = []
rel_err_SVD_lst = []
idx_lst = []
# names = [f"tap={taps}" for taps in numtaps] + ["Nominal"]
names = [f"$\mu$={mus}" for mus in drawn_mus] + ["Nominal"]

fig, ax = plt.subplots(figsize=(8, 6))

err_tol = 5e-2

# mus = [0.01] # only one mu for now

for i in range(len(X_list)):

    X = X_list[i]

    # X_ref is the reference state which is just defined as the mean of the snapshots
    X_ref = np.mean(X, axis=1)[:, None]

    print("X = ", X.shape)
    print("X_ref = ", X_ref.shape)

    # svd
    U, S, V = np.linalg.svd((X - X_ref), full_matrices=False)
    print("S = ", S[:5])
    eigvals_SVD = S**2 * (1 / (len(S) - 1))
    # print("eigvals_SVD = \n", eigvals_SVD[:5])
    # append U
    # print("U = ", U.shape)
    rob_lst.append(U)

    # calculate the relative error
    rel_err_SVD = 1 - (np.cumsum(eigvals_SVD) / np.sum(eigvals_SVD))
    rel_err_SVD_lst.append(rel_err_SVD)
    # print("rel_err_SVD = \n", rel_err_SVD[:4])

    # print the first idx when it is less than 1e-4
    idx = np.where(rel_err_SVD < err_tol)[0][0] + 1
    idx_lst.append(idx)
    print("idx = ", idx)
    print("rel_err_SVD[idx] = ", rel_err_SVD[idx])

    ax.plot(rel_err_SVD_lst[i], label=names[i], linestyle="-", alpha=0.7)
    # ax.plot(idx, rel_err_SVD[idx], 'ro', label=f"{err_tol:.2e} at r={idx}, {names[i]}")
    ax.set_yscale("log")
    ax.set_ylabel("$\\epsilon(r)$")
    # set limit
    # ax.set_xlim([0, 200])
    # ax.set_ylim([1e-13, 1e-3])
    # show grid
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Rank r")
    # ax.set_title("Relative error, RE1000")
    # ax.legend()

# %%
# Model parameters
r = 6  # min for 5e-2
# r = min(idx_lst)
q_trunc = 8
# p = 3

tol = 1e-3  # tolerence for alternating minimization
gamma = 0.01  # regularization parameter\
max_iter = 100  # maximum number of iterations

Vr_lst = []
Vbar_lst = []
Shat_lst = []
Xi_lst = []
Poly_lst = []

for i in range(len(X_list)):
    # Procustes problem for each mu
    X = X_list[i]
    num_snapshots = X.shape[1]
    print("num_snapshots: ", num_snapshots)
    print("X = ", X.shape)
    X_ref = np.mean(X, axis=1)[:, None]
    # X_ref = np.zeros((X.shape[0]))[:, None]
    X_centered = X - X_ref

    U, S, Vr = np.linalg.svd(X_centered, full_matrices=False)

    Vr = U[:, :r]
    Vbar = U[:, r : r + q_trunc]
    q = Vr.T @ X_centered
    Proj_error = X_centered - (Vr @ q)
    Poly = np.concatenate(polynomial_form(q, p), axis=0)
    Xi = (
        Vbar.T
        @ Proj_error
        @ Poly.T
        @ np.linalg.inv(Poly @ Poly.T + gamma * np.identity((p - 1) * r))
    )

    energy = (
        np.linalg.norm(Vr @ q + (Vbar @ Xi @ Poly), "fro") ** 2
        / np.linalg.norm(X - X_ref, "fro") ** 2
    )

    print(f"Snapshot energy: {energy:e}")

    Gamma_MPOD = X_ref + (Vr @ q) + (Vbar @ Xi @ Poly)
    print(f"\nReconstruction error: {relative_error(X, Gamma_MPOD, X_ref):.4%}")

    Vr_lst.append(Vr)
    Vbar_lst.append(Vbar)
    Shat_lst.append(q)
    Xi_lst.append(Xi)
    Poly_lst.append(Poly)

    # q, energy, Xi, Vr, Vbar, Poly = alternating_minimization(X, X_ref, num_snapshots, max_iter, 1e-3, gamma, r, q, p, initial_Shat=None)

    # print("q = ", q.shape)
    # print("qbar = ", qbar.shape)

# %%
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
# Create a figure and a subplot
fig, ax = plt.subplots(figsize=(8, 2))  # Adjust the figure size as needed

# Loop through the colors and plot each one as a rectangle
for i, color in enumerate(color_tags):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.text(
        i + 0.5,
        0.5,
        str(i + 1),
        color="white" if i % 2 == 0 else "black",
        horizontalalignment="center",
        verticalalignment="center",
    )

# Set limits and remove axes for better visualization
ax.set_xlim(0, len(colors))
ax.set_ylim(0, 1)
ax.axis("off")

# Show the plot
plt.show()

# %%
X_all_test[:, 0, None].shape

# %%
dt

# %%
import importlib
import OpInf

# import reloading
# reload the whole OpInf module
importlib.reload(OpInf)
from OpInf import *


config["robparams"] = {"r": int(r)}

operators_lst = []

# use each mu in between the min and max mu
for i in range(len(X_list)):

    X = X_list[i]
    X_ref = np.mean(X, axis=1)[:, None]
    X_centered = X - X_ref

    # U, S, Vr = np.linalg.svd(X_centered, full_matrices=False)

    Vr = Vr_lst[i]
    Vbar = Vbar_lst[i]
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
        100000,
        100000,
        1,
        100000000000,
        100000000000,
        1,
    ]  # r =5, 1e-1

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


# %%
T_end_index = int(T_end_train / dt)

# %% [markdown]
# # Analyze difference in operators

# %%
operators_lst[0].keys()

# %%
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


# %%
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

# %%
# representatives = [105, 51, 77]

# # %%
# from adjustText import adjust_text
# from matplotlib.colors import BoundaryNorm

# # Create a colormap with the same number of colors as clusters
# cmap = plt.get_cmap("rainbow", optimal_n_clusters)

# # Define the boundaries for the discrete color intervals
# boundaries = np.arange(optimal_n_clusters + 1) - 0.5
# norm = BoundaryNorm(boundaries, cmap.N, clip=True)

# fig, ax = plt.subplots(figsize=(8, 6))
# scatter = ax.scatter(
#     V_combined_embedding[:, 0],
#     V_combined_embedding[:, 1],
#     c=cluster_labels,
#     cmap=cmap,
#     norm=norm,
#     alpha=0.7,
# )

# # Create a discrete colorbar
# cbar = fig.colorbar(
#     scatter, ax=ax, boundaries=boundaries, ticks=np.arange(optimal_n_clusters)
# )
# cbar.set_label("Cluster")

# # Set the tick labels for the colorbar
# cbar.set_ticklabels([f"Cluster {i}" for i in range(optimal_n_clusters)])

# # ax.set_title('Spectral Embedding of $[V]$')
# texts = []
# for i, rep in enumerate(representatives):
#     # ax.scatter(V_combined_embedding[rep, 0], V_combined_embedding[rep, 1], c='black', marker='x', s=100)
#     ax.scatter(
#         V_combined_embedding[rep, 0],
#         V_combined_embedding[rep, 1],
#         c=cluster_labels[rep],
#         cmap=cmap,
#         norm=norm,
#         s=100,
#         marker="x",
#         label=names[rep],
#     )
#     text = ax.annotate(
#         f"Rep {i}",
#         (V_combined_embedding[rep, 0], V_combined_embedding[rep, 1]),
#         fontsize=12,
#     )
#     texts.append(text)

# adjust_text(texts, arrowprops=dict(arrowstyle="->", color="red"))
# ax.legend()
# plt.show()

# %% [markdown]
# # get distinct operators and evaluate

# %%
import tqdm

abs_error_full_lst_operators = []
relative_error_testing_window_lst_operators = []
relative_error_training_window_lst_operators = []
s_rec_full_lst_operators = []

# dt_sim = 0.000625

for i in tqdm.tqdm(range(len(X_list) - 1)):

    print("i: ", i)

    operators = operators_lst[i]

    X = X_list[i]
    print("X shape: ", X.shape)
    X_ref = np.mean(X, axis=1)[:, None]
    # X_centered = X - X_ref

    Vr = Vr_lst[i]
    Vbar = Vbar_lst[i]

    q0 = Vr.T @ (X_all_test[:, 0, None] - X_ref).flatten()

    T_end_full = 8
    # time_domain_full = np.arange(t_start, T_end_full, dt_sim)
    time_domain_full = np.arange(t_start, T_end_full, dt)

    multi_indices = generate_multi_indices_efficient(len(q0), p=p)

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
    print("Shape of s_hat_full: ", s_hat_full.shape)

    del out_full
    poly_full = np.concatenate(polynomial_form(s_hat_full, p=p), axis=0)
    # Xi = Xi_lst[-1] # the nominal Xi
    Xi = Xi_lst[i]
    # print("Poly shape: ", poly_full.shape)

    s_rec_full = X_ref + Vr @ s_hat_full + Vbar @ Xi @ poly_full

    try:

        abs_error_full = np.abs(X_all_test - s_rec_full)
        relative_error_testing_window = np.linalg.norm(
            X_all_test[:, T_end_index:] - s_rec_full[:, T_end_index:], "fro"
        ) / np.linalg.norm(X_all_test[:, T_end_index:], "fro")
        relative_error_training_window = np.linalg.norm(
            X_all_test[:, :T_end_index] - s_rec_full[:, :T_end_index], "fro"
        ) / np.linalg.norm(X_all_test[:, :T_end_index], "fro")

        abs_error_full_lst_operators.append(abs_error_full)
        relative_error_testing_window_lst_operators.append(
            relative_error_testing_window
        )
        relative_error_training_window_lst_operators.append(
            relative_error_training_window
        )
        s_rec_full_lst_operators.append(s_rec_full)

    except Exception as e:
        print(e)
        abs_error_full_lst_operators.append(None)
        relative_error_testing_window_lst_operators.append(None)
        relative_error_training_window_lst_operators.append(None)
        s_rec_full_lst_operators.append(None)


# %%
# count number of Nones
nNone = 0
for arr in s_rec_full_lst_operators:
    if arr is None:
        nNone += 1

print(nNone)


# %%
# load mesh coordinates
for mu in mus:
    Re = int(
        np.round(1.0 * 0.1 * 1 / mu)
    )  # Calculate Re based on U_c, L_c, rho, and mu
    folder_name = f"Re_{Re}_mu_{mu:.6f}"
    # mesh_data = np.load(base_path / folder_name / "mesh_data.npz")

    # coordinates = mesh_data["coords"]

    mesh = read_mesh(
        base_path / folder_name / "u_snapshots_mu{mu:.6f}.bp".format(mu=mu)
    )

    V = dolfinx.fem.functionspace(mesh, ("CG", 2))
    coordinates = V.tabulate_dof_coordinates()

    break


# %%
# save Vr_lst, Vbar_lst
with open(
    "/data1/jy384/research/Data/UnimodalSROB/ns/NavierStokes_testing_data.pkl", "wb"
) as f:
    pkl.dump(
        {
            "Vr_lst": Vr_lst,
            "Vbar_lst": Vbar_lst,
            "Xi_lst": Xi_lst,
            "Shat_lst": Shat_lst,
            "Poly_lst": Poly_lst,
            "operators_lst": operators_lst,
            "representatives": representatives,
            # "RiemannKMeans": riem_kmeans,
        },
        f,
    )

print("Done getting data")
