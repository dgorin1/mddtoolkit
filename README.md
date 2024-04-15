# Multiple-Diffusion Domain Model Toolkit (MDD_toolkit) 

This software package is a companion to the paper "Revisiting the MDD Model with Modern Optimization Techniques", and offers tools for understanding the results of stepwise degassing experiments (McDougall and Harrison, 1999) through the lens of the Multi-Diffusion Domain Model (Harrison, 2013). We utilize SciPy's implementation of Differential Evolution in order to search for all MDD-model parameters simultaneously. This software returns plots with the best-fitting model parameters, as well as the parameters themselves. These values can then be used to forward-model thermal histories with other popular programs like Arvert 4.0 (Zeitler., 2004). Advanced users may feel free to experiment with the optimization parameters for customized results, but beginner users should feel comfortable using our software using the default configuration.
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Installation

1. Run `pip install -e .` in the root directory of the repository to install the package and its dependencies

## Usage
### Quick Start

The MDD_toolkit utilizes a command-line interface to configure and begin an optimization run. The command for starting such an optimization is shown below:


```
fit_MDD_model \
    -i path_to/diffusion_code_final/example/tests/N13ksp_python_test.csv \
    -c path_to/diffusion_code_final/example/tests/test.yaml \
    -o /Users/username/repos/diffusion_code_final/output/test_final
```
The pathways required for this command are as follows: \
-i is a file pathway to the input .csv file with the experimental data \
-c is a file pathway to the input .yaml file with settings for the optimizer \
-o is a file pathway for the output.



In order to tune a multi-diffusion domain model to the results of your diffusion experiment, you'll need two files which can be located anywhere on your computer:
1. A .csv with the results of your diffusion experiment formatted like our template (found at: source/example/config.yaml). Do not include headers! Columns for the input need to be supplied in the correct order and are as follows:
    _ i. step number (starting at 0). 
    _ ii. Temperature (°C). 
    _ iii. Duration (hours, non-cumulative). 
    _ iv. Moles measured for each step (moles, non-cumulative). 
    _ v. Uncertainty on each measurement (moles). 


2. A .yaml file which will supply the software with details about the optimization configuration.  We recommend copying and pasting the example we share below and customizing it to your needs if necessary.

example yaml file:
```yaml
##################################################################################
# All users should examine and potentially adjust these settings.

domains_to_model: [1,8] 

lnd0aa_bounds: 
- -5.0
- 50.0

ea_bounds: 
- 50.0
- 500.0

geometry: plane sheet

omit_value_indices: []

misfit_stat_list:
-chisq 
-percent_frac 

##################################################################################
# Advanced settings (Only users with knowledge of the Differential Evolution
# algorithm should adjust these settings)

punish_degas_early: false

repeat_iterations: 10

popsize: 15

seed: 0

tol: 0.00001

max_iters: 100000
```
### Basic Settings

**domains_to_model:** Number of Domains to Model: User should specify this as a range. E.g. [1,8] means to fit a 1, 2, 3, and 4 domain model. [4] specifies that only a 4 domain model is fit.

**lnd0aa_bounds:** This sets the range of values the optimizer can search for the lnd0aa value for each domain in units of ln(1/seconds).

**geometry:** The geometry to be used by all diffusion domains. Currently-supported options are "plane sheet" and "spherical"

**omit_value_indices:** Indicate the values you'd like to omit from the fitting exercise. We use standard python indexing--indexing begins at 0. E.g. if you'd like to omit the 26th heating step in your experiment, you should write "[25]".

**misfit_stat_list:** List either "chisq", "percent_frac", or both in the manner shown below. These are both of the misfit statistics presented in Gorin et al., 2024.

### Advanced Settings

**punish_degas_early:** Some optimization-misfit-statistic combinations may incentivize the model to degas far too early, especially when there are many heating steps at the end of the experiment without much gas. We leave this off by default, but encourage the user to carefully examine the results of their optimzations.

**repeat_iterations:** Number of Times to Repeat Optimization: Because of the stochastic nature of the differential evolution algorithm, we run each optimization 10x by default and return the only the results from the best optimization. Adjust this value if you'd like to increase or decrease this number.

**popsize:** Population Size: This is the number of vectors the differential evoltion algorithm attempts to improve simultaneously. Convergence time decreases as population size decreases, but the likelihood of getting stuck in a local minimum decreases. We have found through trial and error that we produce the best results by repeating optimizations with a population size of ~15 about 10 times instead of making this number higher.

**seed:** Due to the stochastic nature of the algorithm, we set a seed so that you are returned the same results each time you run it. If you'd like to see slightly different results, you may want to alter this value. Values 0 -- 2^32 are accepted.

**tol:** This is the criteria the differential evolution algorithm uses to determine when it has fully converged. We don't think this will be necessary for the average user to adjust. Smaller values typically lead to longer convergence times, while larger values lead to shorter times.


**max_iters:** This is the number of generations the differential evoltion algorithm is allowed to generate before it is forced to return its best-fitting individual. If your optimization run is consistently hitting 100k iterations, you may want to increase this value.





## Output
**input_samplename.csv:** A file containing the calculated diffusivities resulting from the input file.

**N_dom_best_params.pdf:** A pdf with plots showing the results of the optimization.

**combined_results_misfit_type.csv:** A csv file containing the best-fit MDD parameters.

**N_dom_optimizer_output.json:** A less readable, but more detailed, output from the optimizer containing standard metrics about the optimization run.


## Support

Please [open an issue](https://github.com/dgorin1/diffusion_code_final/issues/new) for support or to request new features. Please include as much specific information as possible.

## Contributing

Please contribute to our project by using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/dgorin1/diffusion_code_final/compare/).
