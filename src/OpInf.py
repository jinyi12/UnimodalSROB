import numpy as np
import numpy as np
import scipy
import logging
import scipy.integrate
import scipy.optimize
import warnings
import itertools
import numba
from numba import njit, cfunc
from numba.typed import Dict, List
from numba import types

from scipy.special import comb
from itertools import combinations

from operators import ckron, ckron_indices


def get_x_sq(X):
    """
    Takes data in X where each column is one variable and generates matrix X2,
    where each column contains a quadratic term of the variables in X - this
    is different from a Kronecker product because there is no redundancy.

    Parameters:
    X : array-like, shape (N, w)
        N-by-w data matrix where N is the number of data points, w is the number of variables

    Returns:
    X2 : array-like, shape (N, w*(w+1)/2)
        N-by-(w*(w+1)/2) matrix of quadratic terms of data in X

    Author:
    Adapted from MATLAB code by Elizabeth Qian (elizqian@mit.edu) 17 June 2019
    Python adaptation by ChatGPT
    """
    N, w = X.shape
    columns = []
    for j in range(w):
        # X[:, j] is the j-th column; multiply it by each of its subsequent columns including itself
        for k in range(j, w):
            columns.append(X[:, j] * X[:, k])

    # Stack all the columns horizontally to form the new matrix X2
    X2 = np.column_stack(columns)
    return X2


@numba.njit(parallel=False, fastmath=True)
def gen_poly(X, p, multi_indices=None):
    """
    Compute polynomial terms for a 2D array X transposed, where each row
    is a variable and each column a sample, based on provided multiIndices.
    The output is structured so that each row corresponds to a polynomial
    term and each column to a sample.

    Parameters:
    X (np.ndarray): N-by-M data matrix where N is the number of variables, M is the number of samples.
    p (int): Degree of the polynomial.

    Returns:
    resultArray (np.ndarray): Array containing the computed polynomial terms.
    """
    N, M = X.shape
    if multi_indices is None:
        raise ValueError("Multi-indices must be provided for this function")

    max_degree = 2 * p

    powers = np.ones((N, max_degree + 1, M), dtype=X.dtype)
    for n in range(N):
        for degree in range(1, max_degree + 1):
            powers[n, degree, :] = X[n, :] ** degree

    resultArray = np.ones((multi_indices.shape[0], M), dtype=X.dtype)
    for i, indices in enumerate(multi_indices):
        term = np.ones(M)
        for n in range(N):
            power = indices[n]  # Use directly for Python's zero-based indexing
            if power > 0:  # Check if power is > 0, since 0th degree is already handled
                term *= powers[n, power, :]
        resultArray[i, :] = term

    return resultArray


def generate_multi_indices_efficient(num_vars, p):
    """
    Generate all monomial degrees combinations that satisfy the following conditions:
    1. Sum of indices is between 3 and 2*p inclusive
    2. At most two non-zero entries
    3. No entry is greater than p if there are more than one non-zero entries

    Parameters:
    num_vars (int): Number of variables.
    p (int): Degree of the polynomial.

    Returns:
    validDegrees (np.ndarray): Array of valid degrees.
    """
    max_degree = 2 * p
    degrees = monomial_degrees(num_vars, max_degree)

    # Vectorized filtering criteria:
    # 1. Sum of indices is between 3 and 2*p inclusive
    sum_condition = np.sum(degrees, axis=1) >= 3
    sum_condition &= np.sum(degrees, axis=1) <= 2 * p

    # 2. At most two non-zero entries
    non_zero_condition = np.sum(degrees > 0, axis=1) <= 2

    # 3. No entry is greater than p if there are more than one non-zero entries
    more_than_one_non_zero = np.sum(degrees > 0, axis=1) > 1
    max_condition = (np.max(degrees, axis=1) <= p) | (~more_than_one_non_zero)

    # Combine conditions using element-wise logical AND
    valid_idx = sum_condition & non_zero_condition & max_condition
    valid_degrees = degrees[valid_idx, :]

    # Assuming valid_degrees is a numpy array defined earlier
    row_sums = np.sum(valid_degrees, axis=1)  # Compute row-wise sums

    # Create a tuple of sorting keys with the row sums as the first key,
    # followed by the columns of valid_degrees in reverse order.
    # The row sums need to be the last element of the tuple because lexsort sorts by the last key first.
    keys = tuple(valid_degrees[:, i] for i in range(valid_degrees.shape[1])[::-1]) + (
        row_sums,
    )

    # Perform the sorting using lexsort.
    sort_idx = np.lexsort(keys)

    # Reorder valid_degrees based on the sorting indices.
    valid_degrees = valid_degrees[sort_idx, :]

    return valid_degrees


def monomial_degrees(num_vars, max_degree):
    """
    Generates all monomials up to a given degree for a specified number of variables.

    Parameters:
    num_vars (int): The number of variables.
    max_degree (int): The maximum degree of monomials.

    Returns:
    numpy.ndarray: A matrix where each row represents the degrees of a monomial.
    """
    if num_vars == 1:
        return np.arange(max_degree + 1).reshape(-1, 1)

    degrees = []
    for n in range(max_degree + 1):
        # Generating all combinations of dividers
        dividers = np.array(list(combinations(range(1, n + num_vars), num_vars - 1)))
        if dividers.size == 0:
            dividers = np.array([[0] * (num_vars - 1)])
        else:
            dividers = np.flipud(dividers)

        # Calculating the differences and adjustments to generate degrees
        row_degrees = np.hstack(
            [
                dividers[:, :1],
                np.diff(dividers, axis=1),
                n + num_vars - dividers[:, -1:],
            ]
        )
        row_degrees = row_degrees - 1
        degrees.extend(row_degrees)

    return np.array(degrees)


def calculate_combinatorial(r, p):
    """
    Count the total number of unique monomials.

    Parameters:
    r (int): Number of variables.
    p (int): Degree of the polynomial.

    Returns:
    total_unique_monomials (int): Total number of unique monomials.
    """
    # Count for single-variable monomials
    single_variable_count = r * (2 * p - 2)

    # Initialize count for two-variable monomials
    two_variable_count = 0

    # Calculate two-variable monomial counts
    for d in range(3, 2 * p + 2):
        for i in range(1, min(p, d - 1) + 2):
            if d - i <= p:
                # Choose any two variables out of r for the monomial
                two_variable_count += comb(r, 2, True)

    # Total unique monomials is the sum of single and two-variable counts
    total_unique_monomials = single_variable_count + two_variable_count
    return int(total_unique_monomials)


@numba.njit(fastmath=True, parallel=False)
def tikhonov_poly(b, A, size_params, k1, k2, k3):
    """
    Solves linear regression Ax = b with Tikhonov regularization penalty.

    Parameters:
    b (np.ndarray): Right-hand side.
    A (np.ndarray): Data matrix.
    size_params (dict): Dictionary containing the size of the operators:
        l (int): Number of linear terms.
        s (int): Number of quadratic terms.
        mr (int): Number of bilinear terms.
        m (int): Number of input terms.
        c (int): Number of constant terms.
        drp (int): Number of polynomial terms.
    k1 (float): Tikhonov weighting for the first and second operators (constant and linear terms).
    k2 (float): Tikhonov weighting for the third operator (quadratic term).
    k3 (float): Tikhonov weighting for the fourth operator (polynomial term).

    Returns:
    x (np.ndarray): Solution.
    """
    _, q = b.shape
    _, p = A.shape

    # Create the Tikhonov matrix
    l, s, mr, m, c, drp = size_params.values()

    pseudo = np.eye(p)
    pseudo[:l, :] = np.sqrt(k1) * pseudo[:l, :]
    pseudo[l : l + s, :] = np.sqrt(k2) * pseudo[l : l + s, :]

    if m > 0:
        pseudo[l + s : l + s + m, :] = np.sqrt(k1) * pseudo[l + s : l + s + m, :]
        if c > 0:
            pseudo[l + s + m : l + s + m + c, :] = (
                np.sqrt(k1) * pseudo[l + s + m : l + s + c, :]
            )
            if drp > 0:
                pseudo[l + s + c : l + s + c + drp, :] = (
                    np.sqrt(k3) * pseudo[l + s + c : l + s + c + drp, :]
                )
    else:
        pseudo[l + s : l + s + c, :] = np.sqrt(k1) * pseudo[l + s : l + s + c, :]
        if c > 0:
            pseudo[l + s + c : l + s + c + drp, :] = (
                np.sqrt(k3) * pseudo[l + s + c : l + s + c + drp, :]
            )

    A_plus = np.vstack((A, pseudo))
    b_plus = np.vstack((b, np.zeros((p, q))))

    print("Regularization parameters: ", k1, k2, k3)

    print("Solving...")
    x = np.linalg.lstsq(A_plus, b_plus)[0]
    # l2solver = L2Solver(regularizer=k1)
    # print("Fitting...")
    # l2solver.fit(A_plus, b_plus)
    # x = l2solver.predict()

    return x


def get_modelform(regs):
    """Return the rom_operator_inference ROM modelform that is appropriate for
    the number of regularization parameters (fully quadratic or fully cubic).

    Parameters
    ----------
    regs : two or three non-negative floats
        Regularization hyperparameters for Operator Inference.

    Returns
    -------
    modelform : str
        'cAHB' for fully quadratic ROM; 'cAHGB' for fully cubic ROM.
    """
    if np.isscalar(regs) or len(regs) == 2:
        return "LCQ"
    elif len(regs) == 3:
        return "LCPQ"
    raise ValueError("expected 2 or 3 regularization hyperparameters")


def check_regs(regs):
    """Assure there are the correct number of non-negative regularization
    hyperparameters.

    Parameters
    ----------
    regs : list/ndarray of two or three non-negative floats
        Regularization hyperparameters.
    """
    if np.isscalar(regs):
        regs = [regs]

    # Check number of values.
    nregs = len(regs)
    if nregs not in (2, 3):
        raise ValueError(f"expected 2 or 3 hyperparameters, got {nregs}")

    # Check non-negativity.
    if any(λ < 0 for λ in regs):
        raise ValueError("regularization hyperparameters must be non-negative")

    return regs


def is_bounded(q_rom, B, message="bound exceeded"):
    """Return True if the absolute integrated POD coefficients lie within the
    given bound.

    Parameters
    ----------
    q_rom : (r,len(time_domain)) ndarray
        Integrated POD modes, i.e., the direct result of integrating a ROM.
    B : float > 0
        The bound that the integrated POD coefficients must satisfy.
    """
    if np.abs(q_rom).max() > B:
        print(message + "...", end="")
        logging.info(message)
        return False
    return True


def regularizer(r, λ1, λ2, λ3=None):
    """Return the regularizer that penalizes all operator elements by λ1,
    except for the quadratic operator elements, which are penalized by λ2.
    If λ3 is given, the entries of the cubic operator are penalized by λ3.

    Parameters
    ----------
    r : int
        Dimension of the ROM.
    λ1 : float
        Regularization hyperparameter for the non-quadratic operators.
    λ2 : float
        Regularization hyperparameter for the quadratic operator.
    λ2 : float or None
        Regularization hyperparameter for the cubic operator (if present).

    Returns
    -------
    diag(𝚪) : (d,) ndarray
        Diagonal entries of the dxd regularizer 𝚪.
    """
    r1 = 1 + r
    r2 = r1 + r * (r + 1) // 2
    if λ3 is None:
        diag𝚪 = np.full(r2 + 1, λ1)
        diag𝚪[r1:-1] = λ2
    else:
        r3 = r2 + r * (r + 1) * (r + 2) // 6
        diag𝚪 = np.full(r3 + 1, λ1)
        diag𝚪[r1:r2] = λ2
        diag𝚪[r2:-1] = λ3
    return diag𝚪


import numpy as np
import scipy.optimize
import scipy.integrate
import warnings
import logging


def train_minimize(
    Q_,
    Qdot_,
    Qtrue,
    trainsize,
    r,
    regs,
    time_domain,
    q0,
    params,
    testsize=None,
    margin=1.1,
):
    """
    Train Reduced Order Models (ROMs) using optimization for hyperparameter selection.

    This function trains ROMs with the given dimension, saving only the ROM with
    the least training error that satisfies a bound on the integrated POD
    coefficients. It uses an optimization algorithm to choose the regularization
    hyperparameters.

    Parameters
    ----------
    Q_ : array_like
        The full order model data.
    Qdot_ : array_like
        The time derivative of the full order model data.
    Qtrue : array_like
        The true solution data for error calculation.
    trainsize : int
        Number of snapshots to use for training the ROM.
    r : int
        Dimension of the desired ROM (number of retained POD modes).
    regs : tuple
        Initial guesses for the regularization hyperparameters.
    time_domain : array_like
        Time points for integration.
    q0 : array_like
        Initial condition for integration.
    params : dict
        Dictionary containing model parameters.
    testsize : int, optional
        Number of time steps for which a valid ROM must satisfy the POD bound.
    margin : float, optional
        Allowed deviation factor for integrated POD coefficients (default is 1.1).

    Returns
    -------
    tuple
        Best regularization parameters, operators, and errors, or None if optimization fails.
    """

    _MAXFUN = 100  # Maximum function value for optimization routine

    # Set up model parameters
    modelform = params["modelform"]
    multi_indices = (
        generate_multi_indices_efficient(r, params["p"]) if "P" in modelform else None
    )

    # Convert regularization parameters to log scale
    log10regs = np.log10(check_regs(regs))

    # Compute the bound for integrated POD modes
    B = margin * np.abs(Q_).max()

    print(f"Constructing least-squares solver, r={r}")

    def training_error(log10regs):
        """
        Calculate training error for given regularization parameters.

        Parameters
        ----------
        log10regs : array_like
            Log10 of regularization parameters.

        Returns
        -------
        float
            Training error or _MAXFUN if bound is violated.
        """
        regs = 10**log10regs

        # Update regularization parameters
        params.update({f"lambda{i+1}": reg for i, reg in enumerate(regs)})
        params["lambda3"] = params.get("lambda3", 0)  # Set lambda3 to 0 if not provided

        # Train the ROM
        operators = infer_operators_nl(Q_, None, params, Qdot_)

        # Simulate the ROM
        print("Integrating...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = scipy.integrate.solve_ivp(
                rhs,
                [time_domain[0], time_domain[-1]],
                q0,
                t_eval=time_domain,
                args=[operators, params, None, multi_indices],
            )
        q_rom = out.y

        # Check for boundedness of solution
        if not is_bounded(q_rom, B):
            return _MAXFUN * np.abs(q_rom).max()

        # Calculate and return the error
        error = Lp_error(Qtrue, q_rom, time_domain)[1]
        print(f"Error: {error}")
        print(f"regs: {list(regs)}")
        return error

    # Perform optimization
    opt_result = scipy.optimize.minimize(
        training_error,
        log10regs,
        method="Nelder-Mead",
        tol=1e-9,
        options={"adaptive": True},
    )

    if opt_result.success and opt_result.fun != _MAXFUN:
        # Optimization successful
        best_regs = 10**opt_result.x
        params.update({f"lambda{i+1}": reg for i, reg in enumerate(best_regs)})

        print(f"Best regularization for k={trainsize:d}, r={r}: {list(best_regs)}")

        # Train final model with best parameters
        operators = infer_operators_nl(Q_, None, params, Qdot_)

        # Simulate final model
        q_rom = scipy.integrate.solve_ivp(
            rhs,
            [time_domain[0], time_domain[-1]],
            q0,
            t_eval=time_domain,
            args=[operators, params, None, multi_indices],
        ).y

        # Calculate final errors
        errors = Lp_error(Qtrue, q_rom[:, :trainsize], time_domain[:trainsize])

        return list(best_regs), operators, errors
    else:
        # Optimization failed
        message = "Regularization search optimization FAILED"
        print(message)
        logging.info(message)
        return None


def rhs(t, state, operators, params, input_func=None, multi_indices=None):
    r"""Evaluate the right-hand side of the model by applying each operator
    and summing the results.

    This is the function :math:`\Ophat(\qhat, \u)`
    where the model can be written as one of the following:

    * :math:`\ddt\qhat(t) = \Ophat(\qhat(t), \u(t))` (continuous time)
    * :math:`\qhat_{j+1} = \Ophat(\qhat_j, \u_j)` (discrete time)
    * :math:`\widehat{\mathbf{g}} = \Ophat(\qhat, \u)` (steady state)

    Parameters
    ----------
    state : (r,) ndarray
        State vector.
    input_ : (m,) ndarray or None
        Input vector corresponding to the state.

    Returns
    -------
    evaluation : (r,) ndarray
        Evaluation of the right-hand side of the model.
    """
    modelform = params["modelform"]
    state = np.atleast_1d(state)
    p = params["p"]
    if multi_indices is None and "P" in modelform:
        multi_indices = generate_multi_indices_efficient(state.shape[0], p)

    out = np.zeros(state.shape, dtype=float)

    if "L" in modelform:
        out += operators["A"] @ state

    if "Q" in modelform:
        r, r2 = operators["F"].shape
        mask = ckron_indices(r)
        out += (operators["F"] @ np.prod(state[mask], axis=1)).flatten()

    if "P" in modelform:
        gs = gen_poly(state[:, None], p=p, multi_indices=multi_indices)
        out += (operators["P"] @ gs).flatten()

    if "C" in modelform:
        out += operators["C"].flatten()

    return out


def train_gridsearch(
    Q_,
    Qdot_,
    Qtrue,
    trainsize,
    r,
    regs,
    time_domain,
    q0,
    params,
    testsize=None,
    margin=1.1,
):
    """
    Train Reduced Order Models (ROMs) using grid search for hyperparameter optimization.

    This function trains ROMs with the given dimension, saving only the ROM with
    the least training error that satisfies a bound on the integrated POD
    coefficients. It uses a grid search algorithm to choose the regularization
    hyperparameters.

    Parameters
    ----------
    Q_ : array_like
        The full order model data.
    Qdot_ : array_like
        The time derivative of the full order model data.
    Qtrue : array_like
        The true solution data for error calculation.
    trainsize : int
        Number of snapshots to use for training the ROM.
    r : int
        Dimension of the desired ROM (number of retained POD modes).
    regs : tuple
        Bounds and sizes for the grid of regularization hyperparameters.
        Format: (min1, max1, num1, min2, max2, num2, [min3, max3, num3])
        Where min/max are the bounds and num is the number of points for each regularization parameter.
    time_domain : array_like
        Time points for integration.
    q0 : array_like
        Initial condition for integration.
    params : dict
        Dictionary containing model parameters.
    testsize : int, optional
        Number of time steps for which a valid ROM must satisfy the POD bound.
    margin : float, optional
        Allowed deviation factor for integrated POD coefficients (default is 1.1).

    Returns
    -------
    tuple
        Best regularization parameters and minimum error, or None if no stable ROMs found.
    """

    _MAXFUN = 100  # Maximum function value for optimization routine

    # Validate and parse regularization parameter grids
    if len(regs) not in [6, 9]:
        raise ValueError(
            "Expected 6 or 9 regularization parameters (bounds / sizes of grids)"
        )
    grids = [
        np.logspace(np.log10(regs[i]), np.log10(regs[i + 1]), num=regs[i + 2])
        for i in range(0, len(regs), 3)
    ]

    # Extract model form and generate multi-indices if needed
    modelform = params["modelform"]
    multi_indices = (
        generate_multi_indices_efficient(r, params["p"]) if "P" in modelform else None
    )

    # Compute the bound for integrated POD modes
    B = margin * np.abs(Q_).max()

    # Prepare for grid search
    num_tests = np.prod([grid.size for grid in grids])
    print(f"Constructing least-squares solver, r={r}")
    print(f"TRAINING {num_tests} ROMS")

    errors_pass = {}
    errors_fail = {}

    # Perform grid search
    for i, regs in enumerate(itertools.product(*grids)):
        print(f"({i+1:d}/{num_tests:d}) Testing ROM with {regs}")

        # Set regularization parameters
        params.update({f"lambda{j+1}": reg for j, reg in enumerate(regs)})
        params["lambda3"] = params.get("lambda3", 0)  # Set lambda3 to 0 if not provided

        # Train the ROM
        try:
            operators = infer_operators_nl(Q_, None, params, Qdot_)
        except Exception as e:
            print(f"Operators inference failed: {str(e)}")
            continue

        # Simulate the ROM
        print("Integrating...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = scipy.integrate.solve_ivp(
                rhs,
                [time_domain[0], time_domain[-1]],
                q0,
                t_eval=time_domain,
                vectorized=False,
                args=[operators, params, None, multi_indices],
            )

        if out.status != 0:
            print(f"INTEGRATION FAILED at t = {out.t[-1]}")
            continue

        q_rom = out.y

        # Check if solution is bounded
        if is_bounded(q_rom, B):
            print("Bound check passed")
            # Calculate integrated relative errors in the reduced space
            error_range = min(q_rom.shape[1], trainsize)
            errors_pass[tuple(regs)] = Lp_error(
                Qtrue[:, :error_range],
                q_rom[:, :error_range],
                time_domain[:error_range],
            )[1]
        else:
            print("BOUND EXCEEDED")
            errors_fail[tuple(regs)] = _MAXFUN

    # Select the best ROM
    if not errors_pass:
        message = f"NO STABLE ROMS for r={r:d}"
        print(message)
        logging.info(message)
        return None

    best_regs = min(errors_pass, key=errors_pass.get)
    min_err = errors_pass[best_regs]

    logging.info(f"Best regularization for k={trainsize:d}, r={r:d}: {best_regs}")
    return best_regs, min_err


class L2Solver:
    def __init__(self, regularizer=0):
        """
        Initialize the L2Solver with a regularization parameter.

        Parameters:
        regularizer : float, non-negative
            The regularization parameter λ (lambda) for L2 regularization.
        """
        if not np.isscalar(regularizer) or regularizer < 0:
            raise ValueError("Regularization parameter must be a non-negative scalar")
        self.regularizer = regularizer
        self.is_fitted = False

    def fit(self, A, B):
        """
        Perform SVD on A and prepare for solving the regularized least squares problem.

        Parameters:
        A : ndarray of shape (k, d)
            The "left-hand side" matrix A.
        B : ndarray of shape (k, r)
            The "right-hand side" matrix B, where each column is a target vector.
        """
        self.U, self.s, self.Vt = np.linalg.svd(A, full_matrices=False)
        self.UtB = np.dot(self.U.T, B)
        self.is_fitted = True
        return self

    def predict(self):
        """
        Solve the regularized least squares problem using the precomputed SVD.

        Returns:
        X : ndarray of shape (d, r)
            The solution matrix X to the regularized least squares problem.
        """
        if not self.is_fitted:
            raise RuntimeError("The solver has not been fitted yet.")

        # Calculate the inverse singular values with regularization
        sigma_inv = self.s / (self.s**2 + self.regularizer**2)
        sigma_inv_mat = np.diag(sigma_inv)

        # Compute the solution using the filtered singular values
        X = np.dot(self.Vt.T, np.dot(sigma_inv_mat, self.UtB))
        return X

    def cond(self):
        """
        Compute the 2-norm condition number of the data matrix A.

        Returns:
        condition_number : float
            The condition number of A based on its singular values.
        """
        if not self.is_fitted:
            raise RuntimeError("The solver has not been fitted yet.")

        return self.s.max() / self.s.min()

    def regcond(self):
        """
        Compute the 2-norm condition number of the regularized data matrix.

        Returns:
        reg_condition_number : float
            The condition number of the matrix A with regularization considered.
        """
        if not self.is_fitted:
            raise RuntimeError("The solver has not been fitted yet.")

        regularized_svals = self.s**2 + self.regularizer**2
        return np.sqrt(regularized_svals.max() / regularized_svals.min())

    def residual(self, X):
        """
        Calculate the residual of the regularized problem for each column of B.

        Parameters:
        X : ndarray of shape (d, r)
            The solution matrix X.

        Returns:
        residuals : ndarray of shape (r,)
            The residuals for each solution corresponding to each column of B.
        """
        if not self.is_fitted:
            raise RuntimeError("The solver has not been fitted yet.")

        # Compute residuals for each solution
        Ax = np.dot(self.U, np.dot(np.diag(self.s), np.dot(self.Vt, X)))
        residuals = np.linalg.norm(self.UtB - Ax, axis=0) ** 2 + (
            self.regularizer**2
        ) * np.sum(X**2, axis=0)
        return residuals


def infer_operators_nl(Shat, U, params, rhs=None):
    """
    Infers linear, quadratic, bilinear, input, and constant matrix operators
    for state data in Shat projected onto basis Vr and optional input data U.

    Parameters:
    Shat (np.ndarray): N-by-K full state data matrix.
    U (np.ndarray): K-by-m input data matrix.
    params (dict): Parameters for operator inference:
        modelform (str): Indicates which terms to learn, e.g. 'LI' for linear model with input.
        modeltime (str): 'continuous' or 'discrete'.
        dt (float): Timestep used to calculate state time derivative for continuous-time models.
        lambda1 (float): L2 penalty weighting for constant and linear terms.
        lambda2 (float): L2 penalty weighting for quadratic term.
        lambda3 (float): L2 penalty weighting for polynomial term.
        p (int): Degree of the polynomial.
        scale (bool): If True, scale data matrix to within [-1, 1] before least-squares solve.
    rhs (np.ndarray, optional): N-by-K optional user-specified right-hand side for least-squares solve.

    Returns:
    operators (dict): Inferred operators A, F, N, B, C, P.
    """
    if "modeltime" not in params and rhs is None:
        raise ValueError(
            "Discrete vs continuous not specified and no RHS provided for LS solve."
        )

    if (
        "dt" not in params
        and "modeltime" in params
        and params["modeltime"] == "continuous"
        and rhs is None
    ):
        raise ValueError(
            "No dXdt data provided and no timestep provided in params with which to calculate dXdt"
        )

    if "ddt_order" not in params:
        params["ddt_order"] = 1

    if "lambda1" not in params:
        params["lambda1"] = 0

    if "lambda2" not in params:
        params["lambda2"] = 0

    if "lambda3" not in params:
        params["lambda3"] = 0

    if "p" not in params:
        params["p"] = 2

    if "scale" not in params:
        params["scale"] = False

    ind = np.arange(Shat.shape[1])

    D, size_params = get_data_matrix(Shat, U, ind, params["modelform"], params["p"])

    print("Obtained data matrix...")

    size_params_numba = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64,
    )

    for key, value in size_params.items():
        size_params_numba[key] = value

    temp = tikhonov_poly(
        rhs,
        D,
        size_params_numba,
        params["lambda1"],
        params["lambda2"],
        params["lambda3"],
    ).T
    print("Solved!")

    operators = {
        "A": temp[:, : size_params["l"]],
        "F": temp[:, size_params["l"] : size_params["l"] + size_params["s"]],
        "N": temp[
            :,
            size_params["l"]
            + size_params["s"] : size_params["l"]
            + size_params["s"]
            + size_params["mr"],
        ],
        "B": temp[
            :,
            size_params["l"]
            + size_params["s"]
            + size_params["mr"] : size_params["l"]
            + size_params["s"]
            + size_params["mr"]
            + size_params["m"],
        ],
        "C": temp[
            :,
            size_params["l"]
            + size_params["s"]
            + size_params["mr"]
            + size_params["m"] : size_params["l"]
            + size_params["s"]
            + size_params["mr"]
            + size_params["m"]
            + size_params["c"],
        ],
        "P": temp[
            :,
            size_params["l"]
            + size_params["s"]
            + size_params["mr"]
            + size_params["m"]
            + size_params["c"] : size_params["l"]
            + size_params["s"]
            + size_params["mr"]
            + size_params["m"]
            + size_params["c"]
            + size_params["drp"],
        ],
    }

    return operators


def get_data_matrix(Shat, U, ind, modelform, p):
    """
    Builds the data matrix based on the desired form of the learned model.

    Parameters:
    Shat (np.ndarray): r-by-K state data matrix.
    U (np.ndarray): K-by-m input data matrix.
    ind (np.ndarray): Indices of the state data to use.
    modelform (str): Indicates which terms to learn.
    p (int): Degree of the polynomial.

    Returns:
    D (np.ndarray): Data matrix.
    size_params (dict): Dictionary containing the size of the operators.
    """
    K = len(ind)
    r = Shat.shape[0]

    Shat = Shat[:, ind]
    m = U.shape[1] if U is not None else 0

    if "I" in modelform:
        U0 = U[ind, :]
    else:
        U0 = np.array([])
        m = 0

    if "Q" in modelform:
        # Ssq = get_x_sq(Shat.T)
        Ssq = ckron(Shat).T
        s = r * (r + 1) // 2

    else:
        Ssq = np.array([])
        s = 0

    if "C" in modelform:
        Y = np.ones((K, 1))
        c = 1
    else:
        Y = np.array([])
        c = 0

    if "L" not in modelform:
        Shat = np.array([])
        l = 0
    else:
        l = r

    if "P" in modelform:
        print("Generating ghat ...")

        multi_indices = generate_multi_indices_efficient(num_vars=r, p=p)

        ghat = gen_poly(Shat, p, multi_indices=multi_indices).T

        drp = ghat.shape[1]
        print("drp: ", drp)
    else:
        ghat = np.array([])
        drp = 0

    if "B" in modelform:
        XU = np.hstack([Shat.T * U0[:, i] for i in range(m)])
        mr = m * r
    else:
        # XU = np.array([])[:, None]
        XU = np.empty((K, 0))
        mr = 0

    D = np.hstack([x for x in [Shat.T, Ssq, XU, U0, Y, ghat] if x.size > 0])

    size_params = {
        "l": l,
        "s": s,
        "mr": mr,
        "m": m,
        "c": c,
        "drp": drp,
    }

    return D, size_params


import numpy as np


def semi_implicit_euler_poly(Chat, Ahat, Fhat, Bhat, Phat, dt, u_input, IC, p):

    K = u_input.shape[0]
    r = Ahat.shape[0]
    s_hat = np.zeros((r, K + 1))  # Initial state is zeros everywhere

    s_hat[:, 0] = IC.reshape(-1, 1).flatten()  # Ensure IC is properly shaped

    # Debugging prints omitted for brevity

    multi_indices = generate_multi_indices_efficient(r, p)

    ImdtA = np.eye(r) - dt * Ahat
    for i in range(K):
        # Ensure s_hat[:, i] is reshaped as a column vector for processing
        s_hat_slice = s_hat[:, i].reshape(-1, 1)

        ssq = get_x_sq(s_hat_slice.T).T  # Assuming get_x_sq can handle the shape
        gs = gen_poly(
            s_hat_slice, p=p, multi_indices=multi_indices
        )  # Assuming gen_poly is designed for column vector input

        print(Bhat)

        if Bhat is None or not Bhat:
            update_term = dt * Fhat @ ssq + dt * Phat @ gs + dt * Chat
        else:
            Bhat_term = (
                dt * Bhat @ u_input[i].reshape(-1, 1)
            )  # Ensure u_input[i] is shaped correctly if necessary
            update_term = dt * Fhat @ ssq + Bhat_term + dt * Phat @ gs + dt * Chat

        # Solve and update s_hat[:, i+1]
        s_hat[:, i + 1] = np.linalg.solve(ImdtA, s_hat_slice + update_term).flatten()

        if np.any(np.isnan(s_hat[:, i + 1])):
            print(f"ROM unstable at {i}th timestep")
            break

    return s_hat


def semi_implicit_euler_poly_gen(Chat, Ahat, Fhat, Phat, dt, num_steps, IC, p):

    r = Ahat.shape[0]
    s_hat = np.zeros((r, num_steps + 1))  # Initial state is zeros everywhere

    s_hat[:, 0] = IC.reshape(-1, 1).flatten()  # Ensure IC is properly shaped

    # Debugging prints omitted for brevity

    multi_indices = generate_multi_indices_efficient(r, p)

    ImdtA = np.eye(r) - dt * Ahat
    for i in range(num_steps):
        # Ensure s_hat[:, i] is reshaped as a column vector for processing
        s_hat_slice = s_hat[:, i].reshape(-1, 1)

        ssq = get_x_sq(s_hat_slice.T).T  # Assuming get_x_sq can handle the shape
        gs = gen_poly(
            s_hat_slice, p=p, multi_indices=multi_indices
        )  # Assuming gen_poly is designed for column vector input

        update_term = dt * Fhat @ ssq + dt * Phat @ gs + dt * Chat

        # Solve and update s_hat[:, i+1]
        s_hat[:, i + 1] = np.linalg.solve(ImdtA, s_hat_slice + update_term).flatten()

        if np.any(np.isnan(s_hat[:, i + 1])):
            print(f"ROM unstable at {i}th timestep")
            break

    return s_hat


def semi_implicit_euler(Ahat, Fhat, dt, num_steps, IC):

    r = Ahat.shape[0]
    s_hat = np.zeros((r, num_steps + 1))  # Initial state is zeros everywhere

    s_hat[:, 0] = IC.reshape(-1, 1).flatten()  # Ensure IC is properly shaped

    # Debugging prints omitted for brevity

    ImdtA = np.eye(r) - dt * Ahat
    for i in range(num_steps):
        # Ensure s_hat[:, i] is reshaped as a column vector for processing
        s_hat_slice = s_hat[:, i].reshape(-1, 1)

        ssq = get_x_sq(s_hat_slice.T).T  # Assuming get_x_sq can handle the shape

        update_term = dt * Fhat @ ssq

        # Solve and update s_hat[:, i+1]
        s_hat[:, i + 1] = np.linalg.solve(ImdtA, s_hat_slice + update_term).flatten()

        if np.any(np.isnan(s_hat[:, i + 1])):
            print(f"ROM unstable at {i}th timestep")
            break

    return s_hat


def Lp_error(Qtrue, Qapprox, t=None, p=2):
    """Compute the absolute and relative Lp-norm error (with respect to time)
    between the snapshot sets Qtrue and Qapprox, where Qapprox approximates
    Qtrue:

        absolute_error = ||Qtrue - Qapprox||_{L^p},
        relative_error = ||Qtrue - Qapprox||_{L^p} / ||Qtrue||_{L^p},

    where

        ||Z||_{L^p} = (int_{t} ||z(t)||_{p} dt)^{1/p},          p < infinity,
        ||Z||_{L^p} = sup_{t}||z(t)||_{p},                      p = infinity.

    The trapezoidal rule is used to approximate the integrals (for finite p).
    This error measure is only consistent for data sets where each snapshot
    represents function values, i.e.,

        Qtrue[:, j] = [q(t1), q(t2), ..., q(tk)]^T.

    Parameters
    ----------
    Qtrue : (n, k) or (k,) ndarray
        "True" data corresponding to time t. Each column is one snapshot,
        i.e., Qtrue[:, j] is the data at time t[j]. If one-dimensional, each
        entry is one snapshot.
    Qapprox : (n, k) or (k,) ndarray
        An approximation to Qtrue, i.e., Qapprox[:, j] approximates Qtrue[:, j]
        and corresponds to time t[j]. If one-dimensional, each entry is one
        snapshot.
    t : (k,) ndarray
        Time domain of the data Qtrue and the Qapprox.
        Required unless p == np.inf.
    p : float > 0
        Order of the Lp norm. May be infinite (np.inf).

    Returns
    -------
    abs_err : float
        Absolute error ||Qtrue - Qapprox||_{L^p}.
    rel_err : float
        Relative error ||Qtrue - Qapprox||_{L^p} / ||Qtrue||_{L^p}.
    """
    # Check p.
    if not np.isscalar(p) or p <= 0:
        raise ValueError("norm order p must be positive (np.inf ok)")

    # Check dimensions.
    if Qtrue.shape != Qapprox.shape:
        print("Qtrue shape: ", Qtrue.shape)
        print("Qapprox shape: ", Qapprox.shape)
        raise ValueError("Qtrue and Qapprox not aligned")
    if Qtrue.ndim == 1:
        Qtrue = np.atleast_2d(Qtrue)
        Qapprox = np.atleast_2d(Qapprox)
    elif Qtrue.ndim > 2:
        raise ValueError("Qtrue and Qapprox must be one- or two-dimensional")

    # Pick the norm based on p.
    if 0 < p < np.inf:
        if t is None:
            raise ValueError("time t required for p < infinty")
        if t.ndim != 1:
            raise ValueError("time t must be one-dimensional")
        if Qtrue.shape[-1] != t.shape[0]:
            raise ValueError("Qtrue not aligned with time t")

        def pnorm(Z):
            return (np.trapz(np.sum(np.abs(Z) ** p, axis=0), t)) ** (1 / p)

    else:  # p == np.inf

        def pnorm(Z):
            return np.max(np.abs(Z), axis=0).max()

    # Compute the error.
    return _absolute_and_relative_error(Qtrue, Qapprox, pnorm)


def _absolute_and_relative_error(Qtrue, Qapprox, norm):
    """Compute the absolute and relative errors between Qtrue and Qapprox,
    where Qapprox approximates Qtrue:

        absolute_error = ||Qtrue - Qapprox||,
        relative_error = ||Qtrue - Qapprox|| / ||Qtrue||
                       = absolute_error / ||Qtrue||,

    with ||Q|| defined by norm(Q).
    """
    norm_of_data = norm(Qtrue)
    absolute_error = norm(Qtrue - Qapprox)
    return absolute_error, absolute_error / norm_of_data
