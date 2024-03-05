# Multi-Diffusion Domain Model Toolkit (mddtoolkit) 

This software package is a companion to the paper "Revisiting the MDD Model with Modern Optimization Techniques", and offers a set of tools for understanding the results of step-heating diffusion experiments (McDougall and Harrison, 1999) through the lens of the Multi-Diffusion Domain Model (Harrison, 2013). We utilize SciPy's implementation of Differential Evolution in order to search for all MDD-model parameters simultaneously. This software then returns plots with this fit, as well as the parameters themselves which can then be used for forward modeling with other popular programs. Advanced users may feel free to experiment with the optimization parameters for customized results, but beginner users should feel comfortable using our software without detailed knowledge of the Differential Evolution algorithm.
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Installation

1. Run `pip install -e .` in the root directory of the repository to install the package and its dependencies

## Usage
### Quick Start

In order to tune a multi-diffusion domain model to the results of your diffusion experiment, you'll need two files:
1. A .csv with the results of your diffusion experiment formatted like our template (Found at: source/example/config.yaml). Do not include headers! Columns for the input are as follows: i. step number (starting at 0). ii. Temperature (°C). iii. Duration (hours, non-cumulative). iv. Moles measured for each step (moles, non-cumulative). v. Uncertainty on each measurement (moles). 


2. A .yaml file following the example we share below. We recommend that you copy/paste our example and use it as a template to match your specific needs.

example yaml file:
```yaml
##################################################################################
# All users should examine and potentially these settings

# Number of Domains to Model: User should specify this as a range. E.g. [1,8] means to fit
# a 1, 2, 3, and 4 domain model. [4] specifies that only a 4 domain model is fit.
domains_to_model: [1,4] 

# Lnd0aa Bounds: This sets the range of values the optimizer can search for the 
# lnd0aa value for each domain in units of ln(1/seconds).
lnd0aa_bounds: 
- -5.0
- 50.0

# Activation Energy Bounds: This sets the range of values the optimizer can search
# for the activation energy in units of kJ/mol.
ea_bounds: 
- 50.0
- 500.0

# Geometry: The geometry to be used by all diffusion domains. Currently-supported options 
# are "plane sheet" and "spherical"
geometry: plane sheet

# Omitted Values: Indicate the values you'd like to omit from the fitting exercise. 
# We use standard python indexing--indexing begins at 0. 
# E.g. if you'd like to omit the 26th heating step in your experiment,
# you should write "[25]".
omit_value_indices: []

# Misfit Statistics: List either "chisq", "percent_frac", or both in the manner shown below. 
# These are both of the misfit statistics presented in Gorin (2024).
misfit_stat_list:
-chisq 
-percent_frac 

##################################################################################
# Advanced settings (Only users with knowledge of the Differential Evolution
# algorithm should adjust these settings)

# Degassing-too-early punishment: Some optimization-misfit-statistic combinations 
# may incentivize the model to degas far too early, especially when there are many 
# heating steps at the end of the experiment without much gas. We leave this off by default,
# but encourage the user to carefully examine the results of their optimzations.
punish_degas_early: false

# Number of Times to Repeat Optimization: Because of the stochastic nature of the differential
# evolution algorithm, we run each optimization 10x by default and return the only
# the results from the best optimization. Adjust this value if you'd like to increase 
# or decrease this number
repeat_iterations: 10

# Population Size: This is the number of vectors the differential evoltion algorithm attempts 
# to improve simultaneously. Convergence time decreases as population size decreases, but the 
# likelihood of getting stuck in a local minimum decreases. We have found through trial 
# and error that we produce the best results by repeating optimizations with a population 
# size of ~15 about 10 times instead of making this number higher.
popsize: 15

# Seed: Due to the stochastic nature of the algorithm, we set a seed so that you are returned
# the same results each time you run it. If you'd like to see slightly different results,
# you may want to alter this value. Values 0 -- 2^32 are accepted.
seed: 0

# Tolerance: This is the criteria the differential evolution algorithm uses
# to determine when it has fully converged. We don't think this will be 
# necessary for the average user to adjust. Smaller values typically lead 
# to longer convergence times, while larger values lead to shorter times.
tol: 0.00001



# Maximum allowed iterations: This is the number of generations the 
# differential evoltion algorithm is allowed to generate before it is 
# forced to return its best-fitting individual. If your optimization 
# run is consistently hitting 100k iterations, you may want to 
# increase this value.
max_iters: 100000
```

Once you have your files created and organized, and our software installed, begin your optimization with the following command-line call:
```
fit_MDD_model \
    -i path_to/diffusion_code_final/example/tests/N13ksp_python_test.csv \
    -c path_to/diffusion_code_final/example/tests/test.yaml \
    -o /Users/username/repos/diffusion_code_final/output/test_final
```
 
-i indicates the input .csv file with the experimental data
-c indicates the input .yaml file with settings for the optimizer
-o indicates the output file pathway.




## Support

Please [open an issue](https://github.com/dgorin1/diffusion_code_final/issues/new) for support or to request new features. Please include as much specific information as possible.

## Contributing

Please contribute to our project by using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/dgorin1/diffusion_code_final/compare/).
