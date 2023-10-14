
from diffusion_objective import DiffusionObjective
from jax import numpy as jnp
from dataset import Dataset
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
import random
import time
from plot_results import plot_results
from optimization_routines import diffEV_multiples
from save_results import save_results

# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_input = pd.read_csv(f"{dir_path}/data/input_8DomSynthDataNoisyM3_plane_sheet.csv")
mineral_name = "kspar"
time_add = [] #Add extra time in seconds
temp_add = []
sample_name = "SyntheticArArNoisy_8dom_OmitValues_wPunishment"
moves = "snooker" # Define moves as "snooker" if you fear multimodality in your dataset. Can lead to poor performance if no multimodality exists
max_domains_to_model = 8
geometry  = "plane sheet" #"plane sheet" # options are "plane sheet", or "spherical"
omit_value_indices = [33,34,35,36,37,38,39,40,41] #[]#[0,1,2]#

misfit_stat_list = ["lnd0aa_chisq","chisq","percent_frac","l1_frac_cum","l1_frac","l1_moles","l2_moles","l2_frac","lnd0aa"] # #ADD BACK PERCENT_FRAC. #options are chisq, l1_moles, l2_moles, l1_frac, l2_frac, percent_frac



def organize_x(x,ndim, chop_fracs = True):

        ndom = int(((ndim)/2))
        print(f"ndom is {ndom}")
        if len(x)%2 != 0:

            moles = x[0]
            x = x[1:]
        else:
             moles = np.NaN
        Ea = x[0]
        lnd0aa = x[1:1+ndom]
        fracs = x[1+ndom:]
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

        if "moles" in locals():

            output = np.append(moles,Ea)
        else:
             output = Ea
        output = np.append(output,lnd0aa)
        if chop_fracs == True:
            output = np.append(output,fracs[0:-1])
        else:
             output = np.append(output,fracs)
        return output

# Create dataset class for each associate package

for misfit_stat in misfit_stat_list:

    
    save_params = np.empty((max_domains_to_model-1,max_domains_to_model*2+4))
    save_params.fill(np.NaN)
    for i in range(2,max_domains_to_model+1):
        
        domains_to_model = i
        print(f"{misfit_stat} with {domains_to_model} domains")

        dataset = Dataset("diffEV", data_input)


        objective = DiffusionObjective(
            "diffEV",
            dataset, 
            time_add = torch.tensor(time_add), 
            temp_add = torch.tensor(temp_add), 
            pickle_path = f"{dir_path}/data/lookup_table.pkl",
            omitValueIndices= omit_value_indices,
            stat = misfit_stat,
            geometry = geometry
        )

        # Read in the nonlinear constra¸¸int


        params, misfit_val = diffEV_multiples(objective,dataset,10,mineral_name,domains_to_model)


        plot_results(params,dataset,objective,sample_name=sample_name,quiet = True,misfit_stat = misfit_stat)
        print(sample_name)
        
        params = organize_x(params,len(params),chop_fracs = False)
        print(params)

        if i < max_domains_to_model:
             num_nans_insert = max_domains_to_model-i
             nan_insert = np.empty((num_nans_insert))
             nan_insert.fill(np.NaN)
             array_w_nans = np.insert(params,[2+i],nan_insert,axis=0)
             array_w_nans = np.concatenate((array_w_nans,nan_insert),axis = 0)

        else:
             array_w_nans = params
        add_num_doms = np.append(i,array_w_nans)
        params_to_add = np.append(add_num_doms,misfit_val)

        save_params[i-2,0:len(params_to_add)] = params_to_add


# I need to add a filler if there is no moles
        
        

        save_results(sample_name = sample_name,misfit_stat = misfit_stat,params = save_params)



