import importlib
import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
from numpy.typing import NDArray
from scipy.linalg import svd, logm, expm, qr, eig, expm
from scipy.stats import gamma
from Stiefel_Aux import *
from Stiefel_Exp_Log import Stiefel_Exp, Stiefel_Log

# importlib.reload(Stiefel_Log)
# importlib.reload(Stiefel_Exp)

import jax.numpy as jnp
import jax
from jax import random
from jax import jit


# def stiefel_exp(U0: NDArray, delta: NDArray):

#     """

#     Performs the Stiefel exponential of delta with respect to the reference base point U0.
#     Delta is a tangent vector on TuSt(N,r) \in R^{Nxr}.

#     Parameters:
#     U0: reference base point on St(N,r) \in R^{Nxr}
#     delta: tangent vector on TuSt(N,r) \in R^{Nxr}

#     Returns:
#     U: point on St(N,r) \in R^{Nxr}

#     """

#     A = U0.conj().T @ delta
#     r = U0.shape[1]
#     # thin qr-decomposition
#     Q, R = qr(delta - U0 @ A, mode='economic')
#     # eigenvalue decomposition
#     eig_vals, V = eig(np.block([[A, -R.conj().T], [R, np.zeros((r, r))]]))
#     D = np.diag(eig_vals)
#     MN = V @ expm(D) @ V.conj().T @ np.block([[np.eye(r)], [np.zeros((r, r))]])
#     M = np.real(MN[:r, :])
#     N = np.real(MN[r:, :])
#     U = U0 @ M + Q @ N

#     return U


def stiefel_exp_batch(U0: np.ndarray, delta: np.ndarray, metric_alpha=0.0):
    """
    Performs the Stiefel exponential of delta with respect to the reference base point U0.
    Delta is a batch of tangent vectors on TuSt(N,r) \in R^{batch_size, N, r}.

    Parameters:
    U0: reference base point on St(N,r) \in R^{N, r}
    delta: batch of tangent vectors on TuSt(N,r) \in R^{batch_size, N, r}

    Returns:
    U: batch of points on St(N,r) \in R^{batch_size, N, r}
    """
    batch_size, N, r = delta.shape
    U = np.zeros((batch_size, N, r))

    for i in range(batch_size):
        # A = U0.conj().T @ delta[i]
        # # thin qr-decomposition
        # Q, R = qr(delta[i] - U0 @ A, mode='economic')
        # # eigenvalue decomposition
        # eig_vals, V = eig(np.block([[A, -R.conj().T], [R, np.zeros((r, r))]]))
        # D = np.diag(eig_vals)
        # MN = V @ expm(D) @ V.conj().T @ np.block([[np.eye(r)], [np.zeros((r, r))]])
        # M = np.real(MN[:r, :])
        # N = np.real(MN[r:, :])
        # U[i] = U0 @ M + Q @ N

        U[i] = Stiefel_Exp(U0, delta[i], metric_alpha=metric_alpha)

    return U


# def stiefel_log(U0, U1, tau, verbose=False):
#     """

#     Performs the Stiefel logarithm of U1 with respect to the reference base point U0.
#     The algorithm is guaranteed to converge if U0 and U1 are sufficiently close within 0.09 L2-norm.

#     Parameters:
#     U0: reference base point on St(N,r) \in R^{Nxr}
#     U1: point on St(N,r) \in R^{Nxr}
#     tau: convergence threshold
#     verbose: print convergence information

#     Returns:
#     Delta: tangent vector on TuSt(N,r) \in R^{Nxr}
#     k: number of iterations
#     conv_hist: convergence history
#     norm_logV0: L2-norm of matrix logarithm of V0 which is the orthogonal completion of {{M, N}, {X0, Y0}}.

#     """


#     # get dimensions
#     n, p = U0.shape
#     # store convergence history
#     conv_hist = []
#     # step 1
#     M = U0.conj().T @ U1
#     # step 2, thin qr of normal component of U1
#     Q, N = qr(U1 - U0 @ M, mode='economic')
#     # step 3, orthogonal completion
#     V, _ = qr(np.vstack([M, N]))

#     # "Procrustes preprocessing"
#     D, S, R = svd(V[p:2*p, p:2*p])
#     V[:, p:2*p] = V[:, p:2*p] @ (R @ D.T)
#     V = np.hstack([np.vstack([M, N]), V[:, p:2*p]])

#     # just for the record
#     norm_logV0 = np.linalg.norm(logm(V), 2)

#     # step 4: FOR-Loop
#     for k in range(10000):
#         # step 5
#         LV = logm(V, disp=True)
#         C = LV[p:2*p, p:2*p]   # lower (pxp)-diagonal block
#         C = LV[p:2*p, p:2*p]   # lower (pxp)-diagonal block
#         # steps 6 - 8: convergence check
#         normC = np.linalg.norm(C, 2)
#         conv_hist.append(normC)
#         if normC < tau:
#             if verbose:
#                 print(f'Stiefel log converged after {k} iterations.')
#             break
#         # step 9
#         # Phi = expm(-C) # standard matrix exponential

#         Phi = jax.scipy.linalg.expm(-C)  # jax matrix exponential
#         # step 10
#         V[:, p:2*p] = jax.numpy.matmul(V[:, p:2*p], Phi, precision=jax.lax.Precision.HIGHEST)  # update last p columns


#         print("normC = ", normC)


#     # if loop is not broken, then convergence failed
#     else:
#         print("NormC = ", normC)
#         print('Stiefel log did not converge.')
#     # prepare output
#     Delta = U0 @ LV[:p, :p] + Q @ LV[p:2*p, :p]
#     return Delta, k, conv_hist, norm_logV0


def batch_stiefel_log(U0, rob, tau, metric_alpha=0.0):
    """

    Performs the Stiefel logarithm for each rob with respect to the reference base point U0, the global diffusion map basis.

    Parameters:
    U0: reference base point on St(N,r) \in R^{Nxr}
    rob: list of points on St(N,r) \in R^{Nxr}
    tau: convergence threshold
    verbose: print convergence information

    Returns:
    Deltas: list of tangent vectors on TuSt(N,r) \in R^{Nxr}

    """

    Deltas = []

    for i in range(len(rob)):
        # last rob is the global diffusion map basis
        Delta, conv = Stiefel_Log(U0, rob[i], tau, metric_alpha=metric_alpha)
        Delta = np.real(Delta)
        Deltas.append(Delta)

    Deltas = np.array(Deltas)

    return Deltas


def calc_concen_param(rob: NDArray, Deltas):
    """

    Calculates the concentration parameter beta for the Dirichlet distribution.
    The betas are calculated with respect to the global diffusion map basis.
    The concentration parameter is calculated by solving a convex optimization problem.

    Parameters:
    rob: array of points (matrices) on St(N,r) \in R^{Nxr}
    Deltas: array of tangent vectors (matrices) on TuSt(N,r) \in R^{Nxr}

    """
    num_models = len(rob) - 1
    X = np.reshape(Deltas[:num_models, :, :], (num_models, -1))
    H = X @ X.T
    f = np.zeros(num_models)
    Aeq = np.ones((1, num_models))
    beq = np.array([1])
    lb = np.full(num_models, 1e-15)
    # ub = np.ones(num_models)

    # Define and solve the CVXPY problem
    beta = cp.Variable(num_models)
    # prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(beta, H) + f.T @ beta),
    #                 [Aeq @ beta == beq,
    #                 beta >= lb,
    #                 beta <= ub])

    prob = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(beta, H) + f.T @ beta),
        [Aeq @ beta == beq, beta >= lb],
    )

    prob.solve(eps_abs=1e-10, eps_rel=1e-10, verbose=True)

    # correct beta due to numerical precision
    beta.value = np.maximum(beta.value, 1e-15)

    # Print result
    beta_value = beta.value
    print("beta =", ", ".join(map(str, beta_value)))

    return beta


def gen_tangent_samples(N_samples: int, beta: cp.Variable, X: NDArray, seed: int = 42):
    """

    Generates N_samples tangent samples on TuSt(N,r) \in R^{Nxr} with respect to the global diffusion map basis.
    The tangent samples are generated by sampling from a Dirichlet distribution.

    Parameters:
    N_samples: number of samples
    beta: concentration parameter of the Dirichlet distribution
    X: reshaped array of basis, where X = np.reshape(Deltas[:num_models, :, :], (num_models, -1))

    Returns:
    tangential_samples: array of tangent samples (matrices) on TuSt(N,r) \in R^{Nxr}

    """

    np.random.seed(seed)

    # Generate gamma-distributed random numbers
    scale = 1
    # print(np.tile(beta.value, (N_samples, 1)))
    w = gamma.rvs(
        np.tile(beta.value, (N_samples, 1)),
        scale=scale,
        size=(N_samples, beta.shape[0]),
    )

    # Normalize the rows of w
    w = w / np.sum(w, axis=1, keepdims=True)

    # Compute the tangential samples
    tangential_samples = w @ X

    # # Sort the rows of w and get the indices
    I = np.argsort(w, axis=1)
    maxI = I[:, -1]

    return tangential_samples, maxI


def gen_convex_comb_samples(N_samples: int, beta: NDArray, X: NDArray, seed: int = 42):
    """

    Generates N_samples tangent samples on TuSt(N,r) \in R^{Nxr} with respect to the global diffusion map basis.
    The tangent samples are generated by sampling from a Dirichlet distribution.

    Parameters:
    N_samples: number of samples
    beta: concentration parameter of the Dirichlet distribution
    X: reshaped array of basis, where X = np.reshape(Deltas[:num_models, :, :], (num_models, -1))

    Returns:
    tangential_samples: array of tangent samples (matrices) on TuSt(N,r) \in R^{Nxr}

    """

    np.random.seed(seed)

    # Generate gamma-distributed random numbers
    scale = 1
    # print(np.tile(beta.value, (N_samples, 1)))
    w = gamma.rvs(
        np.tile(beta, (N_samples, 1)),
        scale=scale,
        size=(N_samples, beta.shape[0]),
    )

    # Normalize the rows of w
    w = w / np.sum(w, axis=1, keepdims=True)

    # Compute the tangential samples
    convex_comb_samples = w @ X

    # # Sort the rows of w and get the indices
    I = np.argsort(w, axis=1)
    maxI = I[:, -1]

    return convex_comb_samples, maxI


# def gen_stiefel_samples(N_samples: int, rob: NDArray, tau: float = 1e-4, verbose: bool = True):
def gen_stiefel_samples(
    N_samples: int, rob: NDArray, tau: float = 1e-4, metric_alpha=0.0
):
    """

    Generates N_samples stiefel samples on St(N,r) \in R^{Nxr} with respect to the global diffusion map basis.

    Parameters:
    N_samples: number of samples
    rob: list of points on St(N,r) \in R^{Nxr}
    tau: convergence threshold of the Stiefel logarithm
    verbose: print convergence information

    Returns:
    stiefel_samples: array of stiefel samples (matrices) on St(N,r) \in R^{Nxr}

    """

    # the global ROB as reference base point
    U0 = rob[-1]

    # number of models excluding the global ROB
    num_models = len(rob) - 1

    # rob has shape (num_models, n_points, n), where n_points is the number of points and n is the number of eigenvectors (order of samples)
    n_points = rob[0].shape[0]
    n = rob[0].shape[1]

    # get the tangent vectors deltas
    # Deltas = batch_stiefel_log(U0, rob, tau=tau, verbose=verbose)
    Deltas = batch_stiefel_log(U0, rob, tau=tau, metric_alpha=metric_alpha)

    # calculate the concentration parameter beta
    beta = calc_concen_param(rob, Deltas)

    # reshape Deltas for computing the tangential samples
    X = np.reshape(Deltas[:num_models, :, :], (num_models, -1))

    # compute tangential samples
    tangential_samples, maxI = gen_tangent_samples(N_samples, beta, X)

    # Reshape tangential_samples
    tangential_samples = np.reshape(tangential_samples, (N_samples, n_points, n))

    # Initialize stiefel_samples
    stiefel_samples = np.zeros(tangential_samples.shape)

    # Compute stiefel_samples
    for i in range(N_samples):
        delta = tangential_samples[i, :, :]
        stiefel_samples[i, :, :] = Stiefel_Exp(
            rob[-1], delta, metric_alpha=metric_alpha
        )

    return stiefel_samples, maxI


def calc_frechet_mean_mat(samples, U0, eps, tau=1e-3):
    """

    Calculates the Frechet mean of a set of samples on St(N,r) \in R^{Nxr} with respect to the reference base point U0.
    Subject to the convergence threshold OF THE MEAN CALCULATION eps, the algorithm is guaranteed to converge.

    Parameters:
    samples: array of samples (matrices) on St(N,r) \in R^{Nxr}
    U0: reference base point on St(N,r) \in R^{Nxr}
    eps: convergence threshold of the Frechet mean calculation
    tau: convergence threshold of the Stiefel logarithm

    Returns:
    frechet_mean: Frechet mean of the samples (matrix) on St(N,r) \in R^{Nxr}

    """
    err = np.inf
    c = 1
    N_samples, d, r = samples.shape
    count = 0
    errs = []

    while err > eps:
        V_mean = np.zeros((d, r))
        for i in range(N_samples):
            Delta, conv = Stiefel_Log(U0, samples[i, :, :], tau)
            V_mean = V_mean + np.real(Delta)
        V_mean = c * V_mean / N_samples
        U0 = Stiefel_Exp(U0, V_mean)
        # calculate error
        err = np.linalg.norm(V_mean, "fro")
        errs.append(err)
        count += 1
        print(f"count = {count}, error = {err}")
    frechet_mean = U0
    return frechet_mean, errs
