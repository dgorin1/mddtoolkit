import random
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution
from generate_bounds import generate_bounds
from torch import torch
from dataset import Dataset
from diffusion_objective import DiffusionObjective
import numpy as np
from conHe_Param import conHe_Param


def diffEV_multiples(objective,dataset, num_iters: int, mineral: str ,ndom):
    nlc = NonlinearConstraint(conHe_Param,lb =[0,-np.inf],ub = [np.inf,np.inf])
    misfits = []
    params = []
    seed = random.randint(0,2^28)
    mole_bound = tuple((sum(dataset.M)- 1*torch.sqrt(sum(torch.tensor(dataset.delM) **2)), sum(dataset.M) + 1*torch.sqrt(sum(torch.tensor(dataset.delM) **2))))
    bounds = generate_bounds(ndom, mole_bound, mineral, objective.stat)

    for i in range(num_iters):

        result = differential_evolution(
        objective, 
        bounds, 
        disp=False, 
        tol=0.0001, # zeros seems like a good number from testing. slow, but useful.
        maxiter = 30000,
        constraints = nlc,
        vectorized = True,
        updating = "deferred",
        seed = seed

    )

        misfits.append(result.fun)
        print(f"misfit: {result.fun}")
        print(f"number of iterations: {result.nit}")
        params.append(result.x)
        
        seed += 1
   
    index = np.argmin(misfits)
    return params[index],  misfits[index]