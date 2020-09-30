# A Python interface to the DDalphaAMG multigrid solver library

[![python](https://img.shields.io/pypi/pyversions/lyncs_DDalphaAMG.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_DDalphaAMG/)
[![pypi](https://img.shields.io/pypi/v/lyncs_DDalphaAMG.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_DDalphaAMG/)
[![license](https://img.shields.io/github/license/Lyncs-API/lyncs.DDalphaAMG?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.DDalphaAMG/blob/master/LICENSE)
[![build & test](https://img.shields.io/github/workflow/status/Lyncs-API/lyncs.DDalphaAMG/build%20&%20test?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.DDalphaAMG/actions)
[![codecov](https://img.shields.io/codecov/c/github/Lyncs-API/lyncs.DDalphaAMG?logo=codecov&logoColor=white)](https://codecov.io/gh/Lyncs-API/lyncs.DDalphaAMG)
[![pylint](https://img.shields.io/badge/pylint%20score-9.5%2F10-green?logo=python&logoColor=white)](http://pylint.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=codefactor&logoColor=white)](https://github.com/ambv/black)

This package provides a Python interface to DDalphaAMG.
[DDalphaAMG] is a solver library for inverting Wilson Clover and Twisted Mass fermions from lattice QCD.
It provides an implementation of an adaptive aggregation-based algebraic multigrid ($\alpha$AMG) method.

[DDalphaAMG]: https://github.com/sbacchio/DDalphaAMG

## Installation

**NOTE**: lyncs_DDalphaAMG requires a working MPI installation.
This can be installed via `apt-get`:

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

OR using `conda`:

```bash
conda install -c anaconda mpi4py
```

The package can be installed via `pip`:

```bash
pip install [--user] lyncs_DDalphaAMG
```

## Documentation

The functions provided by the [DDalphaAMG API](https://github.com/sbacchio/DDalphaAMG/blob/master/src/DDalphaAMG.h) are available in the Solver class `lyncs_DDalphaAMG.Solver`.
Please, use `help(Solver)` to see an overview of the functionalities.
In the following we present some examples on the usage of the package.


```python
from lyncs_DDalphaAMG import Solver

# Creating the solver
solver = Solver(global_lattice=[4, 4, 4, 4],
       	 	kappa=0.125)

# Reading the configurations
conf = solver.read_configuration("test/conf.random")
plaq = solver.set_configuration(conf)
print("Plaquette:", plaq)

# Computing the solution of a random vector
vector = solver.random()
result = solver.solve(vector)
```

The above example is run locally and all the fields are numpy arrays.

The Solver class is also Dask compatible and by giving a distributed communicator,
as in the following example, the solver instance is now distributed remotely on
the dask workers and all the fields are Dask arrays.

```python
from lyncs_mpi import Client

# Creating a client with 4 workers
client = Client(num_workers = 4)
comm = client.create_comm()
procs = [2, 2, 1, 1]
comm = comms.create_cart(procs)

solver = Solver(global_lattice=[4, 4, 4, 4],
       	 	comm = comm,
       	 	kappa=0.125)
		
# Continues as above
```
