
from diffusion_objective import DiffusionObjective

from jax import numpy as jnp
from dataset import Dataset
from plot_results import plot_results
import torch as torch
import pandas as pd
import os
import numpy as np
from conHe_Param import conHe_Param
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint
from emcee_main import emcee_main
from generate_bounds import generate_bounds

import random

# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_input = pd.read_csv(f"{dir_path}/data/input_KM95-28-Dc-1250um.csv")
domains_to_model = 6
mineral_name = "quartz"
time_add = [300*60,110073600]
temp_add = [40,21.111111111111]
sample_name = "KM95-28-Dc-1250um"
moves = "default" # Define moves as "snooker" if you fear multimodality in your dataset. Can lead to poor performance if no multimodality exists




def diff_callback(xk,convergence):
    global threshold
    global objective_diffEV
    val = objective_diffEV.objective(xk)
    if val < threshold:
        return True
    else:
        return False

def check_bounds(x,bounds):

    for i in range(len(x)):
        if bounds[i][0] < x[i] < bounds[i][1]:
            return True
    return False

def organize_x(x,ndim):
        ndom = int(((ndim-1)/2))
        moles = x[0]
        Ea = x[1]
        lnd0aa = x[2:2+ndom]
        fracs = x[2+ndom:]
        fracs = np.append(fracs,1-np.sum(fracs))
        
        n = len(fracs)
        # Traverse through all array elements
        for i in range(n):
            
            # Last i elements are already in place
            for j in range(0, n - i - 1):
                
                # Traverse the array from 0 to n-i-1
                # Swap if the element found is greater than the next element
                if lnd0aa[j] < lnd0aa[j + 1]:
                    lnd0aa[j], lnd0aa[j + 1] = lnd0aa[j + 1], lnd0aa[j]
                    fracs[j], fracs[j + 1] = fracs[j + 1], fracs[j]
        output = np.append(moles,Ea)
        output = np.append(output,lnd0aa)
        output = np.append(output,fracs[0:-1])
        return output

# Create dataset class for each associate package
dataset = Dataset("optimize", data_input)
datasetEv = Dataset("diffEv", data_input)

# Create an objective class for each associated package
objective_optimize = DiffusionObjective(
    "ipopt",
    dataset, 
    time_add = jnp.array(time_add), 
    temp_add = jnp.array(temp_add), 
    pickle_path = f"{dir_path}/data/lookup_table.pkl",
    omitValueIndices= []
)

objective_diffEV = DiffusionObjective(
    "diffEv",
    datasetEv, 
    time_add = torch.tensor(time_add), 
    temp_add = torch.tensor(temp_add), 
    pickle_path = f"{dir_path}/data/lookup_table.pkl",
    omitValueIndices= []
)

# Read in the nonlinear constraint
nlc = NonlinearConstraint(conHe_Param,lb =[0,0],ub = [np.inf,np.inf])

mole_bound = tuple((sum(dataset.M)- 1*jnp.sqrt(sum(dataset.delM **2)), sum(dataset.M) + 1*jnp.sqrt(sum(dataset.delM **2))))

bounds = generate_bounds(domains_to_model,mole_bound,"Quartz")
threshold = objective_diffEV.plateau*0.5
seed = random.randint(1, 2**28)
currentMisfit = np.inf
while threshold < currentMisfit:
        result1 = differential_evolution(
            objective_diffEV, 
            bounds, 
            disp=True,
            constraints = nlc,
            callback = diff_callback,
            seed = seed,
            vectorized = True
        )
        currentMisfit = result1.fun

        problem = DiffusionProblem(objective_optimize, bounds) 
        problem.add_option('mu_strategy', 'adaptive')
        problem.add_option('check_derivatives_for_naninf', 'yes')
        problem.add_option('max_iter', 3000)
        problem.add_option('linear_solver', 'mumps')
        seed +=1
        print(seed)
        x0 = jnp.array(result1.x)
        params, info = problem.solve(x0)
        params = organize_x(params,len(params))

        result2 = differential_evolution(
            objective_diffEV, 
            bounds, 
            disp=True, 
            tol=0.0001, #4 zeros seems like a good number from testing. slow, but useful.
            maxiter = 30000,
            constraints = nlc,
            x0 = params,
            #popsize = 100,
            seed = seed,
            vectorized = True
        )
    



print(result2.x)

temp = emcee_main(result2.x, objective_diffEV,sample_name = sample_name,num_iters = 500000,moves= moves)


params_final = []
for i in range(len(temp)):
     params_final.append(temp[i][1])


plot_results(params_final,dataset,objective_optimize,sample_name=sample_name,moves_type = moves)


print(temp)




