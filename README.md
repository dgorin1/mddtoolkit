# Multi-Diffusion Domain Model Toolkit (mddtoolkit) 

This app does things.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Installation

1. Run `pip install -e .` in the root directory of the repository to install the package and its dependencies
2. Dance

## Usage
### Quick Start
example cli call:
```
fit_MDD_model \
    -i path_to/diffusion_code_final/example/tests/N13ksp_python_test.csv \
    -c path_to/diffusion_code_final/example/tests/test.yaml \
    -o /Users/josh/repos/ diffusion_code_final/output/test_final
```

example yaml file:
```yaml
domains_to_model: [1,8]

lnd0aa_bounds:
- -5.0
- 50.0

ea_bounds:
- 50.0
- 500.0

geometry: plane sheet
omit_value_indices: []
punish_degas_early: true 
```

## Support

Please [open an issue](https://github.com/dgorin1/diffusion_code_final/issues/new) for support.

## Contributing

Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/dgorin1/diffusion_code_final/compare/).
