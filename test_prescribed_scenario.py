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
    23,
)  # The user should really be able to pick any numerical values they want here (Good to list its in units of 1/s)
Ea_bounds = (
    50,
    500,
)  # User should also be able to pick any number >0 for these bounds. List in kJ/mol
# mineral_name = "kspar"



# time_add..  C: 111801600   E: 115603200   F: 136944000   G: 139276800    H: 140400000
time_add = [3600*5, 140400000]# Add extra time in seconds
temp_add = [40,21.11111]# Add extra time in degrees C
sample_name = "KM95-15-Dc_TEST" # Sample name
max_domains_to_model = 8
geometry = "spherical"  # options are "plane sheet", or "spherical". Spherical should be default.
omit_value_indices = [0, 1,
]  # Values you want to be ignored in your fit
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
iteration_repeats = 2  # Default should be 10, but user can set to any integer 1-?
punish_degas_early = True #Default is true. Title for gui can be punish if modeled experiment fully degasses too early.


# Create dataset class for each associate package

i = 8
counter = 0
save_params = np.empty((max_domains_to_model - i+1, max_domains_to_model * 2 + 4))
save_params.fill(np.NaN)
prev_misfit = 11**17
misfit_val = 10**17


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


#DC Kinetics 
params = [2822135345.27,	88.96252324607280000,	17.91566513155140000,	17.75631633851090000,	15.59229338494230000,	14.07629577722490000,	11.77379761169650000,	9.16302823096994000	,7.25695164393476000,	-4.73521424225423000,	0.88306462080072800	,0.11649153805080000,	0.00025124762132678,	0.00008152127874705	,0.00000834396748234	,0.00000901068397335	,0.00000549237913322]


params = [372608405.510795, 8.670562585941110000E+01,	1.536971676947920000E+01,	1.357732460714640000E+01,	1.146033407082310000E+01,	8.771864436294710000E+00,	6.821648033277210000E+00, 6.310656042267430000E-01,	2.930717928931170000E-01,	2.422501907973750000E-02,	2.989465672962160000E-02]
# Dc chi sq no correction, with punishment on
#params = [2.84255539754848E+09	,8.68831919285363E+01,	1.46744590541716E+01,	1.37520462942979E+01	,1.32078072353585E+01	,1.13677658162167E+01	,1.01492997016324E+01	,8.68694811004490E+00	,7.21812703513610E+00,	5.80267441495275E+00	,3.86434657314767E-01,	2.48386520980943E-01	,2.01592499578338E-01	,4.45056525795999E-02	,1.43154649406997E-02,	5.38796093388347E-02,3.2895340254633400E-02]

#params = [2.82222070e+09, 9.23233459e+01, 1.56339787e+01, 1.69579750e+01, 8.58050111e+00, 1.47440973e+01, 1.26985593e+01, 1.04812005e+01, 3.19415844e+01, 1.90483985e+01 ,5.30412896e-04 ,1.45107560e-03, 6.40315358e-05, 2.68031209e-04, 5.17934691e-05, 4.82087335e-05, 2.41393509e-02]

# DC_8_only. Note that 7 domains actually fit better.
#params = [2.822143149855370E+09,	8.864694993577250E+01,	1.780236864861670E+01	,1.771265900153830E+01,	1.571505648531430E+01	,1.442991414988730E+01,	1.350130878373850E+01,	1.137678550146280E+01,	8.837846452889490E+00,	6.696526057584820E+00	,7.042593444945320E-01	,2.953812275034270E-01,	2.305166362930870E-04,	8.515162477701340E-05,	2.434762197239810E-05	,6.503168153981770E-06,	8.631063696085120E-06]
#2.8221353452724800E+09	8.8962523246072800E+01	1.7915665131551400E+01	1.7756316338510900E+01	1.5592293384942300E+01	1.4076295777224900E+01	1.1773797611696500E+01	9.1630282309699400E+00	7.2569516439347600E+00	-4.7352142422542300E+00	8.8306462080072800E-01	1.1649153805080000E-01	2.5124762132677700E-04	8.1521278747054200E-05	8.3439674823382300E-06	9.0106839733539100E-06	5.4923791332206400E-06]

# De Kinetics
#params = [1621195750.6318100000000000,	84.7715711226748000,	15.6591214667326000,	14.0630184268234000,	13.0506873052120000,	12.0925095657850000,	9.8235130125182200,	7.8832556366724900,	6.0923343964430200,	4.9160080248146400,	0.9060916409497030,	0.0425248067853096,	0.0355392588458081,	0.0100979003770234,	0.0011160699780227,	0.0029538152784228,	0.0012130345216808]

# Df Kinetics

#params = [2.7282266405780E+09	,8.8846573126701E+01,	1.7669350396377E+01,	1.7064786187979E+01,	1.4868902922513E+01,	1.3412633696655E+01,	1.0696715897754E+01,	8.4270910096391E+00,	6.4776515068929E+00,	5.4777220485977E+00,	4.3497660672063E-02	,9.2446886327642E-01	,1.7339612368701E-02	,3.3629040995881E-03	,1.9124171724202E-03,1.4332076390544E-03	,3.1995990616940E-03]


# Dg kinetics
#params = [1.5937989545921700E+08	,8.7807597742761200E+01,	4.6962316956468400E+01,	2.1062417979933300E+01	,1.6266650880617900E+01	,1.4774542554149800E+01	,1.3265114531406600E+01	,1.1278064392688500E+01	,9.8763735292101000E+00	,8.2726376821228600E+00	,2.1125382708684600E-02	,2.1787860382653200E-02	,7.5841901528216900E-01,	1.7686052187028000E-01	,1.7314589874394000E-02	,1.6661562701276700E-03,	1.3346942623909100E-03]


# Dh Kinetics
#params = [3.72905201776844E+08,	8.89243351233824E+01,	1.71830254749492E+01	,1.52829369495004E+01,	1.40591548112567E+01	,1.20353863963884E+01,	9.88318881428274E+00,	8.35342363489107E+00,	7.08729644246954E+00,	-1.82641225197342E+00,	9.53488975943163E-01,	1.75703164732512E-02,	2.10164565923418E-02,	1.96337302370369E-03,	2.09504292585133E-03,	1.96330769549291E-03,	1.47283124482133E-03]



params = organize_x(params, len(params), chop_fracs=True)

plot_results(
    params,
    dataset,
    objective,
    sample_name=sample_name,
    quiet=False,
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