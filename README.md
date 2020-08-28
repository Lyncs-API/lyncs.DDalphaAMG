# A Python interface to DDalphaAMG

[![python](https://img.shields.io/pypi/pyversions/lyncs_DDalphaAMG.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_DDalphaAMG/)
[![pypi](https://img.shields.io/pypi/v/lyncs_DDalphaAMG.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_DDalphaAMG/)
[![license](https://img.shields.io/github/license/Lyncs-API/lyncs.DDalphaAMG?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.DDalphaAMG/blob/master/LICENSE)
[![build & test](https://img.shields.io/github/workflow/status/Lyncs-API/lyncs.DDalphaAMG/build%20&%20test?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.DDalphaAMG/actions)
[![codecov](https://img.shields.io/codecov/c/github/Lyncs-API/lyncs.DDalphaAMG?logo=codecov&logoColor=white)](https://codecov.io/gh/Lyncs-API/lyncs.DDalphaAMG)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=codefactor&logoColor=white)](https://github.com/ambv/black)

This package provides a Python interface to DDalphaAMG.
[DDalphaAMG] is a solver library for inverting Wilson Clover and Twisted Mass fermions from lattice QCD.
It provides an implementation of an adaptive aggregation-based algebraic multigrid ($\alpha$AMG) method.

## Installation

**NOTE**: lyncs_DDalphaAMG requires a working MPI installation.
This can be installed via `apt-get`:

```
sudo apt-get install libopenmpi-dev openmpi-bin
```

OR using `conda`:

```
conda install -c anaconda mpi4py
```

The package can be installed via `pip`:

```
pip install [--user] lyncs_DDalphaAMG
```

## Documentation

