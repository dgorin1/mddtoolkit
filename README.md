# Multi-Diffusion Domain Model Toolkit (mddtoolkit) 

This app does things.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Installation

1. Run `pip install -e .` in the root directory of the repository to install the package and its dependencies

## Usage
### Quick Start

In order to tune an MDD model to the results of your diffusion experiment, you'll need two files:
1. A .csv with the results of your experiment formatted like our template (source/example/config.yaml).
2. A .yaml file following the example we share below.

example yaml file:
```yaml
domains_to_model: [1,8]

lnd0aa_bounds: # Same for all domains
- -5.0
- 50.0

ea_bounds: # Same for all domains
- 50.0
- 500.0

# Below, indicate the diffusion geometry you'd like to use. Currently-supported options 
# are "plane sheet" and "spherical"
geometry: plane sheet

# Indicate the values you'd like to omit from the fitting exercise. 
# We use standard python indexing here--indexing begins at 0. 
# E.g. if you'd like to omit the 26th heating step in your experiment,
# you should write "[25]".
omit_value_indices: []

# Some optimization-misfit-statistic combinations may incentivize the
# model to degas far too early, especially when there are many heating steps
# at the end of the experiment without much gas. 
punish_degas_early: true 
```

Once you have your files created and organized, and our software installed, use the following command-line interface call:
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

Please [open an issue](https://github.com/dgorin1/diffusion_code_final/issues/new) for support. Include as much specific information as possible

## Contributing

Please contribute to our project by using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/dgorin1/diffusion_code_final/compare/).
