from lyncs_DDalphaAMG import Solver


def test_init():
    solver = Solver(
        global_lattice=[4, 4, 4, 4], block_lattice=[2, 2, 2, 2], procs=[1, 1, 1, 1]
    )
