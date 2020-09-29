from dask.base import wait
from lyncs_mpi import Client
from lyncs_DDalphaAMG import Solver

client = Client(2)
comms = client.create_comm()
comms = comms.create_cart([2, 1, 1, 1])


def test_read_serial(benchmark):
    solver = Solver(global_lattice=[4, 4, 4, 4])
    benchmark(solver.read_configuration, "test/conf.random")


def test_read_parallel(benchmark):
    solver = Solver(global_lattice=[4, 4, 4, 4], comm=comms)
    benchmark(lambda conf: wait(solver.read_configuration(conf)), "test/conf.random")


def test_solve_serial(benchmark):
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        kappa=0.15,
    )

    conf = solver.read_configuration("test/conf.random")
    plaq = solver.set_configuration(conf)
    vec = solver.random()
    sol = benchmark(solver.solve, vec)


def test_solve_parallel(benchmark):
    solver = Solver(
        global_lattice=[4, 4, 4, 4],
        kappa=0.15,
        comm=comms,
    )

    conf = solver.read_configuration("test/conf.random")
    plaq = solver.set_configuration(conf)
    vec = solver.random()
    sol = benchmark(lambda vec: wait(solver.solve(vec)), vec)
