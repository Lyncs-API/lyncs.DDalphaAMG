from pytest import raises
from lyncs_mpi import Client
from lyncs_DDalphaAMG import Solver
import numpy as np
from mpi4py import MPI
from distributed import wait
from dask.array import allclose


def test_properties():
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        block_lattice=[2, 2, 2, 2],
        procs=[1, 1, 1, 1],
        kappa=0.15,
    )
    assert solver.nlevels == 1
    assert solver.global_lattice == (4, 4, 4, 4)
    assert solver.block_lattice == (2, 2, 2, 2)
    assert solver.procs == (1, 1, 1, 1)
    assert solver.local_lattice == (4, 4, 4, 4)

    assert "number_of_levels" in dir(solver)


def test_serial():
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        block_lattice=[2, 2, 2, 2],
        procs=[1, 1, 1, 1],
        kappa=0.15,
    )

    assert isinstance(Solver.comm.fget(solver), MPI.Cartcomm)
    assert isinstance(solver.comm, MPI.Cartcomm)
    assert solver.coords == (0, 0, 0, 0)

    conf = solver.read_configuration("test/conf.random")
    plaq = solver.set_configuration(conf)
    assert np.isclose(plaq, 0.13324460568521923)

    vec = solver.random()
    sol = solver.solve(vec)
    assert np.allclose(solver.D(sol), vec)

    solver.update_setup()
    assert np.allclose(solver.solve(vec), sol)

    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        kappa=0.15,
    )
    plaq = solver.set_configuration(conf)
    assert solver.block_lattice == (2, 2, 2, 2)
    assert np.isclose(plaq, 0.13324460568521923)
    assert np.allclose(solver.D(sol), vec)
    assert np.allclose(solver.solve(vec), sol)


def test_boundary():
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        kappa=0.15,
        boundary_conditions=1,
    )

    conf = solver.read_configuration("test/conf.random")
    plaq = solver.set_configuration(conf)
    assert np.isclose(plaq, 0.13324460568521923)

    vec = solver.random()
    sol = solver.solve(vec)
    assert np.allclose(solver.D(sol), vec)

    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        kappa=0.15,
        boundary_conditions=[0, 0, 0, 0],
    )
    plaq = solver.set_configuration(conf)
    assert solver.block_lattice == (2, 2, 2, 2)
    assert np.isclose(plaq, 0.13324460568521923)
    assert np.allclose(solver.solve(vec), sol)


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


def test_rnd_seeds():
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        rnd_seeds=[
            1234,
        ],
    )
    # rnd_seeds do not really work... everytime different result
    # TODO: fix!
    assert not np.isclose(solver.random().sum(), 0)


def test_errors():
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
    )

    with raises(RuntimeError):
        solver.number_of_levels = 2

    with raises(AttributeError):
        solver.foo = "bar"

    with raises(RuntimeError):
        solver.setup()

    with raises(TypeError):
        Solver(global_lattice=[4, 4, 4, 4], boundary_conditions="periodic")

    with raises(ValueError):
        Solver(
            global_lattice=[4, 4, 4],
        )

    with raises(ValueError):
        Solver(global_lattice=[4, 4, 4, 4], block_lattice=[2, 2])

    with raises(ValueError):
        Solver(global_lattice=[4, 4, 4, 4], block_lattice=[3, 3, 3, 3])

    with raises(TypeError):
        Solver(
            global_lattice=[4, 4, 4, 4],
            comm="foo",
        )

    with raises(ValueError):
        Solver(global_lattice=[4, 4, 4, 4], procs=[2, 2])

    with raises(TypeError):
        Solver(global_lattice=[4, 4, 4, 4], rnd_seeds=10)

    with raises(ValueError):
        Solver(global_lattice=[4, 4, 4, 4], rnd_seeds=[1, 2])


def test_data_movement():
    client = Client(4)
    procs = [2, 1, 1, 1]
    comm1 = client.create_comm(2).create_cart(procs)
    comm2 = client.create_comm(2, exclude=comm1.workers).create_cart(procs)

    solver1 = Solver(
        global_lattice=[4, 4, 4, 4],
        block_lattice=[2, 2, 2, 2],
        procs=procs,
        kappa=0.1,
        comm=comm1,
    )
    solver2 = Solver(
        global_lattice=[4, 4, 4, 4],
        block_lattice=[2, 2, 2, 2],
        procs=procs,
        kappa=0.1,
        comm=comm2,
    )

    conf = solver1.read_configuration("test/conf.random")
    plaq1 = solver1.set_configuration(conf)
    assert np.isclose(plaq1, 0.13324460568521923)

    plaq2 = solver2.set_configuration(conf)
    assert np.isclose(plaq1, plaq2)

    vec = solver1.random()
    sol = solver1.solve(vec)
    assert allclose(solver2.D(sol), vec)
