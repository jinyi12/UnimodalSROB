import numpy as np

def ckron(state, checkdim=False):
        r"""Calculate the compressed Kronecker product of a vector with itself.

        For a vector :math:`\qhat = [~\hat{q}_{1}~~\cdots~~\hat{q}_{r}~]\trp`,
        the Kronecker product of :math:`\qhat` with itself is given by

        .. math::
           \qhat \otimes \qhat
           = \left[\begin{array}{c}
               \hat{q}_{1}\qhat
               \\ \vdots \\
               \hat{q}_{r}\qhat
           \end{array}\right]
           =
           \left[\begin{array}{c}
               \hat{q}_{1}^{2} \\
               \hat{q}_{1}\hat{q}_{2} \\
               \vdots \\
               \hat{q}_{1}\hat{q}_{r} \\
               \hat{q}_{1}\hat{q}_{2} \\
               \hat{q}_{2}^{2} \\
               \vdots \\
               \hat{q}_{2}\hat{q}_{r} \\
               \vdots
               \hat{q}_{r}^{2}
           \end{array}\right] \in\RR^{r^2}.

        Cross terms :math:`\hat{q}_i \hat{q}_j` for :math:`i \neq j` appear
        twice in :math:`\qhat\otimes\qhat`.
        The *compressed Kronecker product* :math:`\qhat\hat{\otimes}\qhat`
        consists of the unique terms of :math:`\qhat\otimes\qhat`:

        .. math::
           \qhat\hat{\otimes}\qhat
           = \left[\begin{array}{c}
               \hat{q}_{1}^2
               \\
               \hat{q}_{2}\qhat_{1:2}
               \\ \vdots \\
               \hat{q}_{r}\qhat_{1:r}
           \end{array}\right]
           = \left[\begin{array}{c}
               \hat{q}_{1}^2 \\
               \hat{q}_{1}\hat{q}_{2} \\ \hat{q}_{2}^{2} \\
               \\ \vdots \\ \hline
               \hat{q}_{1}\hat{q}_{r} \\ \hat{q}_{2}\hat{q}_{r}
               \\ \vdots \\ \hat{q}_{r}^{2}
           \end{array}\right]
           \in \RR^{r(r+1)/2},
           \qquad
           \qhat_{1:i}
           = \left[\begin{array}{c}
               \hat{q}_{1} \\ \vdots \\ \hat{q}_{i}
           \end{array}\right]
           \in\RR^{i}.

        For matrices, the product is computed columnwise:

        .. math::
           \left[\begin{array}{c|c|c}
               & & \\
               \qhat_0 & \cdots & \qhat_{k-1}
               \\ & &
           \end{array}\right]
           \hat{\otimes}
           \left[\begin{array}{ccc}
               & & \\
               \qhat_0 & \cdots & \qhat_{k-1}
               \\ & &
           \end{array}\right]
           = \left[\begin{array}{ccc}
               & & \\
               \qhat_0\hat{\otimes}\qhat_0
               & \cdots &
               \qhat_{k-1}\hat{\otimes}\qhat_{k-1}
               \\ & &
           \end{array}\right]
           \in \RR^{r(r+1)/2 \times k}.

        Parameters
        ----------
        state : (r,) or (r, k) numpy.ndarray
            State vector or matrix where each column is a state vector.

        Returns
        -------
        product : (r(r+1)/2,) or (r(r+1)/2, k) ndarray
            The compressed Kronecker product of ``state`` with itself.
        """
        return np.concatenate(
            [state[i] * state[: i + 1] for i in range(state.shape[0])],
            axis=0,
        )
        
        
def ckron_indices(r):
    """Construct a mask for efficiently computing the compressed Kronecker
    product.

    This method provides a faster way to evaluate :meth:`ckron`
    when the state dimension ``r`` is known *a priori*.

    Parameters
    ----------
    r : int
        State dimension.

    Returns
    -------
    mask : ndarray
        Compressed Kronecker product mask.

    Examples
    --------
    >>> from opinf.operators import QuadraticOperator
    >>> r = 20
    >>> mask = QuadraticOperator.ckron_indices(r)
    >>> q = np.random.random(r)
    >>> np.allclose(QuadraticOperator.ckron(q), np.prod(q[mask], axis=1))
    True
    """
    mask = np.zeros((r * (r + 1) // 2, 2), dtype=int)
    count = 0
    for i in range(r):
        for j in range(i + 1):
            mask[count, :] = (i, j)
            count += 1
    return mask