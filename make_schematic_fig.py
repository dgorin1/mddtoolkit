from optimization.diffusion_objective import DiffusionObjective
from input_data.dataset import Dataset
import torch as torch
import pandas as pd
import os
import numpy as np
from utils.plot_results import plot_results
from optimization.optimization_routines import diffEV_multiples
from utils.save_results import save_results
from utils.organize_x import organize_x

# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))

data_input = pd.read_csv(f"{dir_path}/data/GR27_feldspar_in.csv")
lnd0aa_bounds = (
    -5,
    50,
)  # The user should really be able to pick any numerical values they want here (Good to list its in units of 1/s)
Ea_bounds = (
    50,
    500,
)  # User should also be able to pick any number >0 for these bounds. List in kJ/mol
# mineral_name = "kspar"
time_add = []  # Add extra time in seconds
temp_add = [] # Add extra time in degrees C
sample_name = "first_python_test_cancellingindices" # Sample name
max_domains_to_model = 8
geometry = "plane sheet"  # options are "plane sheet", or "spherical". Spherical should be default.
omit_value_indices = [
]  # Values you want to be ignored in your fit
misfit_stat_list = [
    
     "chisq",
    #  "l1_frac",
    #"lnd0aa_chisq"
    # "percent_frac",
    # "l1_frac_cum",
    # "l1_frac",
    # "l1_moles",
    # "l2_moles",
    # "l2_frac",
    # "lnd0aa",
]  # This is a list of all the options. The user should just pick one.

# Create dataset class for each associate package



dataset = Dataset("diffEV", data_input)
misfit_stat = "chisq"
objective = DiffusionObjective(
    dataset,
    time_add=torch.tensor(time_add),
    temp_add=torch.tensor(temp_add),
    omitValueIndices=omit_value_indices,
    stat=misfit_stat,
    geometry=geometry,
    punish_degas_early = True
)


params = organize_x(params, len(params), chop_fracs=True)
plot_results(
    params,
    dataset,
    objective,
    sample_name=sample_name,
    quiet=True,
    misfit_stat=misfit_stat,
)
print(sample_name)

params = organize_x(params, len(params), chop_fracs=False)
print(params)

if i < max_domains_to_model:
    num_nans_insert = max_domains_to_model - i
    nan_insert = np.empty((num_nans_insert))
    nan_insert.fill(np.NaN)
    array_w_nans = np.insert(params, [2 + i], nan_insert, axis=0)
    array_w_nans = np.concatenate((array_w_nans, nan_insert), axis=0)

else:
    array_w_nans = params
add_num_doms = np.append(i, array_w_nans)
params_to_add = np.append(add_num_doms, misfit_val)
save_params[i-1, 0 : len(params_to_add)] = params_to_add


