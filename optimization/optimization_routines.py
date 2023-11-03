import random
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution
from utils.generate_bounds import generate_bounds
from torch import torch
from input_data.dataset import Dataset
from optimization.diffusion_objective import DiffusionObjective
import numpy as np
from optimization.conHe_Param import conHe_Param


def diffEV_multiples(
    objective,
    dataset,
    num_iters: int,
    ndom,
    Ea_bounds: tuple,
    lnd0aa_bounds: tuple,
    max_iters: int = 30000,
):
    """
    Run the differential evolution algorithm multiple times and returns the best result.

    Args:
        objective (DiffusionObjective): the objective function for the optimization problem.
        dataset (Dataset): the dataset for the optimization problem.
        num_iters (int): the number of times to run the differential evolution algorithm.
        ndom (int): the number of domains in the optimization problem.
        Ea_bounds (tuple): the bounds for the activation energy.
        lnd0aa_bounds (tuple): the bounds for the ln(D0/a^2).
        max_iters (int, optional): the maximum number of iterations for the differential evolution algorithm. Defaults to 30000.

    Returns:
        _type_: _description_
    """
    # If the number of domains > 1, enforce constraing that 1-sum(fracs) > 0. Else, no constraint.
    if ndom > 1:
        nlc = NonlinearConstraint(conHe_Param, lb=[0], ub=[np.inf])
    else:
        nlc = []

    misfits = []
    params = []
    seed = random.randint(0, 2 ^ 28)
    mole_bound = tuple(
        (
            sum(dataset.M) - 1 * torch.sqrt(sum(torch.tensor(dataset.delM) ** 2)),
            sum(dataset.M) + 1 * torch.sqrt(sum(torch.tensor(dataset.delM) ** 2)),
        )
    )
    bounds = generate_bounds(
        ndom,
        mole_bound,
        objective.stat,
        Ea_bounds=Ea_bounds,
        lnd0aa_bounds=lnd0aa_bounds,
    )

    for i in range(num_iters):
        result = differential_evolution(
            objective,
            bounds,
            disp=True,
            tol=0.0001,  
            maxiter=max_iters,
            constraints=nlc,
            vectorized=True,
            updating="deferred",
            seed=seed,

        )

        misfits.append(result.fun)
        print(f"misfit: {result.fun}")
        print(f"number of iterations: {result.nit}")
        params.append(result.x)

        seed += 1

    index = np.argmin(misfits)
    return params[index], misfits[index]
