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

data_input = pd.read_csv(f"{dir_path}/data/input_N13ksp_fwd_bad.csv")
lnd0aa_bounds = (
    -5,
    60,
)  # The user should really be able to pick any numerical values they want here (Good to list its in units of 1/s)
Ea_bounds = (
    50,
    500,
)  # User should also be able to pick any number >0 for these bounds. List in kJ/mol
# mineral_name = "kspar"
time_add = []  # Add extra time in seconds
temp_add = [] # Add extra time in degrees C
sample_name = "ignoreTest" # Sample name
max_domains_to_model = 10
geometry = "plane sheet"  # options are "plane sheet", or "spherical". Spherical should be default.
omit_value_indices = [
]  # Values you want to be ignored in your fit
misfit_stat_list = [
    
     "chisq",
     "l1_frac",
    #"lnd0aa_chisq"
    # "percent_frac",
    # "l1_frac_cum",
    # "l1_frac",
    # "l1_moles",
    # "l2_moles",
    # "l2_frac",
    # "lnd0aa",
]  # This is a list of all the options. The user should just pick one.
max_iters = 100000  # Often 30k is enough, but not always.
iteration_repeats = 2  # Default should be 10, but user can set to any integer 1-?
punish_degas_early = True #Default is true. Title for gui can be punish if modeled experiment fully degasses too early.

x0 = [5.2129E-12, 200.219193288,23.8, 19.4483046687238, 16.1765962791235,	13.9300726859056,	8.89366400206864,	8.67665088325817,	6.81289980055765,	6.17580077979784,0.01,0.025006122630720,0.095924851822800,0.081527591431440,0.223989792642000,0.125357818148640,0.270276835862880,]

# Create dataset class for each associate package

for misfit_stat in misfit_stat_list:
    i = 8
    save_params = np.empty((max_domains_to_model - i+1, max_domains_to_model * 2 + 4))
    save_params.fill(np.NaN)
    prev_misfit = 11**17
    misfit_val = 10**17
    while i < max_domains_to_model + 1 and misfit_val < prev_misfit:
        prev_misfit = misfit_val

        domains_to_model = i
        print(f"{misfit_stat} with {domains_to_model} domains")

        dataset = Dataset("diffEV", data_input)

        objective = DiffusionObjective(
            dataset,
            time_add=torch.tensor(time_add),
            temp_add=torch.tensor(temp_add),
            omitValueIndices=omit_value_indices,
            stat=misfit_stat,
            geometry=geometry,
            punish_degas_early = True
        
        )

        # Read in the nonlinear constra¸¸int

        params, misfit_val = diffEV_multiples(
            objective,
            dataset,
            iteration_repeats,
            domains_to_model,
            Ea_bounds=Ea_bounds,
            lnd0aa_bounds=lnd0aa_bounds,
            max_iters=max_iters,
            x0 = x0
        )

        params = organize_x(params, len(params), chop_fracs=True)
        

        params = organize_x(params, len(params), chop_fracs=False)
        print(params)

        