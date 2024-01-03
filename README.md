# Diffusion Modeler

This app does things.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Installation

1. Clone this repository with `git clone https://github.com/dgorin1/diffusion_code_final`
2. Run `pip install -e .` in the root directory of the repository to install the package and its dependencies

## Usage
example cli call:
```
fit_MDD_model \
    -i path_to/diffusion_code_final/example/tests/N13ksp_python_test.csv \
    -c path_to/diffusion_code_final/example/tests/test.yaml \
    -o /Users/josh/repos/ diffusion_code_final/output/test_final
```

example yaml file:
```yaml
ea_bounds:
- 50.0
- 500.0
geometry: plane sheet
lnd0aa_bounds:
- -5.0
- 60.0
domains_to_model: 8 # number of domains to model
misfit_stat_list:   # different misfit statistics to try
- chisq
- l1_frac
- lnd0aa_chisq
- percent_frac
- l1_frac_cum
- l1_moles
- l2_moles
- l2_frac
- lnd0aa
omit_value_indices: []
punish_degas_early: true
temp_add: []
time_add: []

# optimization parameters
repeat_iterations: 10 # number of times to repeat the optimization per domain/misfit statistic
seed: 0
tol: 0.0001
popsize: 15
updating: deferred
strategy: best1bin
mutation: 0.5
recombination: 0.7
max_iters: 100000
```

## Support

Please [open an issue](https://github.com/dgorin1/diffusion_code_final/issues/new) for support.

## Contributing

Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/dgorin1/diffusion_code_final/compare/).
