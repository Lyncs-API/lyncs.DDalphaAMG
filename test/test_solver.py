from lyncs_mpi import Client
from lyncs_DDalphaAMG import Solver
import numpy as np
from distributed import wait


def test_init():
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        block_lattice=[2, 2, 2, 2],
        procs=[1, 1, 1, 1],
        kappa=0.1,
    )

    conf = solver.read_configuration("test/conf.random")
    plaq = solver.set_configuration(conf)
    assert plaq == 0.13324460568521923

    vec = np.zeros([4, 4, 4, 4, 4, 3])
    vec[0, 0, 0, 0, 0, 0] = 1
    sol = solver.solve(vec)
    assert np.allclose(solver.D(sol), vec)


def test_mpi():
    client = Client(2)
    comms = client.create_comm()

    procs = [2, 1, 1, 1]
    comms = comms.create_cart(procs)
    create_solver = lambda comm: Solver(
        global_lattice=[4, 4, 4, 4],
        block_lattice=[2, 2, 2, 2],
        procs=[1, 1, 1, 1],
        kappa=0.1,
        comm=comm,
    )
    solver = client.map(create_solver, comms)
    assert len(solver) == 2
    wait(solver)


def test_parallel():
    client = Client(2)
    comms = client.create_comm()

    procs = [2, 1, 1, 1]
    comms = comms.create_cart(procs)
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        block_lattice=[2, 2, 2, 2],
        procs=[1, 1, 1, 1],
        kappa=0.1,
        comm=comms,
    )
    assert len(solver) == 2
