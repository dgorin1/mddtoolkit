import torch as torch
import numpy as np


def conHe_Param(X: torch.Tensor):  # Input is Ea, lnd0aa_x, Frac_x-1
    """
    Calculate the gas release from a single heating step using the Arrhenius equation and the Fechtig and Kalbitzer model for gas release.

    Args:
        X (torch.Tensor): A tensor containing the Ea, lnD0aa_x, and Frac_x-1 for each domain.

    Returns:
        torch.Tensor: A tensor containing the constraints for the optimization problem.
    """
    X = X[1:]
    # CONSTRAING 1: FRACTIONS MUST ADD TO 1
    # Determine number of domains
    if len(X) <= 3:
        ndom = 1
    else:
        ndom = (len(X)) // 2
    temp = X[1:]
    fracstemp = temp[ndom:]  # fracstemp has numDomain-1 entries 1-sum(Fracstemp)
    frac_id = 1 - sum(fracstemp)

    # CONSTRAINT 2: lnD0aa1 > lnD0aa2.... > lnD0aa_n

    lnD0aa = temp[0:ndom]  # Has shape (1,num_domains)

    lnD0aa_diff = lnD0aa[:-1] - lnD0aa[1:]
    lnD0aa_check = np.sum(
        np.where(lnD0aa_diff <= 0, lnD0aa_diff, np.array(0.0)), axis=0
    )

    return np.array([frac_id, lnD0aa_check])

    # constraint 1 must be >0
    # Constraint 2 must be between negative infinity and zero
