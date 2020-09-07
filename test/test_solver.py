from lyncs_DDalphaAMG import Solver
import numpy as np


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
