from optimization.diffusion_objective import DiffusionObjective
from input_data.dataset import Dataset
import torch as torch
import pandas as pd
import os
import numpy as np
from utils.plot_results_schematic import plot_results_schematic
from optimization.optimization_routines import diffEV_multiples
from utils.save_results import save_results
from utils.organize_x import organize_x

# get this file's directory
dir_path = os.path.dirname(os.path.realpath(__file__))

data_input = pd.read_csv(f"{dir_path}/input_GR27_feldspar_in.csv")
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
sample_name = "GR27_few" # Sample name
max_domains_to_model = 8
geometry = "plane sheet"  # options are "plane sheet", or "spherical". Spherical should be default.
omit_value_indices = [32,33,34,35,36,37,38
]  # Values you want to be ignored in your fit


dataset = Dataset("diffEV", data_input)

misfit_stat = "chisq"
objective = DiffusionObjective(
    dataset,
    time_add=torch.tensor(time_add),
    temp_add=torch.tensor(temp_add),
    omitValueIndices=omit_value_indices,
    stat=misfit_stat,
    geometry=geometry,
    punish_degas_early = False
)


params = [3.91457517581200E+02,	4.62200768821121E+01,	3.90180663316288E+01	,3.28200751498657E+01,	2.72285786410262E+01	,2.24478990017536E+01	,6.93795393437542E+00,	4.81875188588035E+00,	2.22805911996708E-01,	6.39612889665484E-02,	9.76777710994787E-02,	3.73290912608320E-02	,1.98939822739980E-01,	3.52387260186070E-01,	7.29078123887070E-02,	4.66218454753439E-02,	1.30175107883039E-01]
# 1.73365444341471E-13,	is the first panel
params = organize_x(params, len(params), chop_fracs=False)
plot_results_schematic(
    params,
    dataset,
    objective,
    plot_path = dir_path+"/schematic_fig"
)



print(params)

