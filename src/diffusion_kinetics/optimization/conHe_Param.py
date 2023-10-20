import torch as torch
import numpy as np


def conHe_Param(X: torch.Tensor):  # Input is Ea, lnd0aa_x, Frac_x-1
    """
    Determines if the potential entry to the optimizer satisfies nonlinear constraint that the sum of the 
    input fractions must be less than zero. This is because the sum of all gas fractions must all add up to 1.
    Therefore, if there are N domains, then the gas fraction of the Nth domain is determined by the sum of the 
    preceeding N-1. Therefore, if the sum of the preceeding is greater than 1, we have violated the constraint.

    Args:
        - X (torch.Tensor): A tensor containing the total moles (if using a misfit statistic 
        that calculates this), Ea, lnD0aa_x, and Frac_x-1 for each domain
        

    Returns:
        - torch.Tensor: A tensor containing the constraints for the optimization problem.
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

    return np.array([frac_id])

