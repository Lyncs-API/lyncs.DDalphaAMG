from lyncs_mpi import Client
from lyncs_DDalphaAMG import Solver
import numpy as np
from distributed import wait
from dask.array import allclose


def test_serial():
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        block_lattice=[2, 2, 2, 2],
        procs=[1, 1, 1, 1],
        kappa=0.1,
    )

    assert solver.coords == (0, 0, 0, 0)

    conf = solver.read_configuration("test/conf.random")
    plaq = solver.set_configuration(conf)
    assert np.isclose(plaq, 0.13324460568521923)

    vec = solver.random()
    sol = solver.solve(vec)
    assert np.allclose(solver.D(sol), vec)


def test_parallel():
    client = Client(2)
    comms = client.create_comm()

    procs = [2, 1, 1, 1]
    comms = comms.create_cart(procs)
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        block_lattice=[2, 2, 2, 2],
        procs=procs,
        kappa=0.1,
        comm=comms,
    )
    assert len(solver) == 2

    conf = solver.read_configuration("test/conf.random")
    plaq = solver.set_configuration(conf)
    assert np.isclose(plaq, 0.13324460568521923)

    vec = solver.random()
    sol = solver.solve(vec)
    assert allclose(solver.D(sol), vec)
