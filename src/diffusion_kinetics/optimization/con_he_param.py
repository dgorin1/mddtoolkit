import numpy as np


def con_he_param(X: np.ndarray) -> np.ndarray:
    """Nonlinear constraint: the gas fractions of all N domains must sum to ≤ 1.

    Because the Nth domain fraction is implicitly ``1 - sum(frac_1 … frac_{N-1})``,
    the optimizer only carries the first ``N-1`` fractions as free parameters.
    This function returns that implicit last fraction; the optimizer enforces
    it to be ≥ 0 (via ``lb=[0]``), which is equivalent to requiring the sum
    of all fractions to be ≤ 1.

    Args:
        X: Full parameter vector ``[Ea, lnD0/a²_1, ..., frac_1, ..., frac_{N-1}]``.
            For the chisq statistic the leading ``total_moles`` entry is included
            and is skipped here via ``X[1:]``.

    Returns:
        np.ndarray: Single-element array containing the value of the last
        (implicit) domain fraction.
    """
    X = X[1:]  # skip total_moles (chisq) or Ea — handled upstream
    n_dom = len(X) // 2
    fracs_temp = X[1:][n_dom:]  # the N-1 explicit fractions
    return np.array([1 - sum(fracs_temp)])
