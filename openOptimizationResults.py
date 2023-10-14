
from diffusion_objective import DiffusionObjective
from diffusion_problem import DiffusionProblem
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
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_input = pd.read_csv(f"{dir_path}/data/input_KM95-28-Dc-1250um.csv")
domains_to_model = 4
mineral_name = "quartz"
time_add = [300*60,110073600]
temp_add = [40,21.111111111111]




# Create dataset class for each associate package
dataset = Dataset("optimize", data_input)
datasetEv = Dataset("diffEv", data_input)

# Create an objective class for each associated package
objective_optimize = DiffusionObjective(
    "optimize",
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



with open('results_diffEV_pop15_mumps_KM95-28-4Dom.pkl', 'rb') as f:
    # Read the pickle file
    data = pickle.load(f)


with open('tuner_trials_KM95-28_5dom.pkl', 'rb') as f:
    # Read the pickle file
    tune_results = pickle.load(f)

breakpoint()


misfits = []
params = []
for i in range(len(data)):
    params.append(data[i].x)
    misfits.append(data[i].fun)

misfits = np.array(misfits)

min_index = np.argmin(np.array([misfits]))

#starting_params = params[min_index]
# starting_params = params[9]

# print(starting_params)
# # breakpoint()
# # plt.figure()
# # sns.histplot(data=misfits[misfits<12000])
# # plt.show()
# breakpoint()
# plot_results(starting_params,dataset,objective_diffEV)



