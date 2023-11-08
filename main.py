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

data_input = pd.read_csv(f"{dir_path}/data/input_KM95-15-Dh.csv")
lnd0aa_bounds = (
    -5,
    50,
)  # The user should really be able to pick any numerical values they want here (Good to list its in units of 1/s)
Ea_bounds = (
    50,
    500,
)  # User should also be able to pick any number >0 for these bounds. List in kJ/mol
# mineral_name = "kspar"
time_add = [3600*5,111801600]# Add extra time in seconds
temp_add = [40,21.11111]# Add extra time in degrees C
sample_name = "KM95-15-Dh_CorrectButNoPunishment" # Sample name
max_domains_to_model = 8
geometry = "spherical"  # options are "plane sheet", or "spherical". Spherical should be default.
omit_value_indices = []  #For Dc I used [0,1,2,3,4,5]. For De, I used [0,1,2,3,4]. For Df, I used [0,1,2,3,4,5,6]. For Dg, I used [0,1,2]. For Dh, I used [0,1,2] Values you want to be ignored in your fit
misfit_stat_list = [
    "chisq",
    "percent_frac",
    "l1_frac_cum",
    "l1_frac",
    "l1_moles",
    "lnd0aa_chisq",
    "lnd0aa",

]  # This is a list of all the options. The user should just pick one.
max_iters = 100000 # Often 30k is enough, but not always.
iteration_repeats = 10  # Default should be 10, but user can set to any integer 1-?
punish_degas_early = False #Default is true. Title for gui can be punish if modeled experiment fully degasses too early.


# Create dataset class for each associate package

for misfit_stat in misfit_stat_list:
    i = 1
    counter = 0
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

        save_params[counter, 0 : len(params_to_add)] = params_to_add

        save_results(
            sample_name=sample_name, misfit_stat=misfit_stat, params=save_params
        )
        misfit = misfit_val
        i += 1
        counter += 1
