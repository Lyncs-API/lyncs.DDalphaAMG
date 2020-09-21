"""
Interface to the DDalphaAMG solver library.

Reference C header and documentation can be found in
https://github.com/sbacchio/DDalphaAMG/blob/master/src/DDalphaAMG.h
"""

__all__ = [
    "Solver",
]

import logging
from array import array
from os.path import isfile, realpath, abspath
import numpy
from cppyy import nullptr
from mpi4py import MPI
from lyncs_mpi import default_comm, CartesianClass
from lyncs_mpi.abc import Array, Global
from lyncs_cppyy.ll import cast, to_pointer, addressof
from lyncs_utils import factors, prime_factors, compute_property, isiterable
from . import lib
from .config import WITH_CLIME


class Solver(metaclass=CartesianClass):
    """
    The DDalphaAMG solver class.
    """

    initialized = False
    __slots__ = [
        "_init_params",
        "_run_params",
        "_status",
        "updated",
        "_setup",
        "_local_lattice",
    ]

    def __init__(
        self,
        global_lattice=None,
        block_lattice=None,
        procs=None,
        comm=None,
        boundary_conditions=-1,
        number_of_levels=1,
        number_openmp_threads=1,
        init_file=None,
        rnd_seeds=None,
        **kwargs,
    ):
        """
        Initialize a new DDalphaAMG solver class.

        Parameters
        ----------
        global_lattice: int[4]
            Size of the lattice. The directions order is T, Z, Y, X.
        block_lattice: int[4]
            Size of the first level blocking. The directions order is T, Z, Y, X.
        procs: int[4]
            Number of processes per direction. The directions order is T, Z, Y, X.
        comm: MPI.Comm
            It can be (a) MPI_COMM_WORLD, (b) A split of MPI_COMM_WORLD,
            (c) A cartesian communicator with 4 dims and number of processes in
            each directions equal to procs[4] and with proper boundary conditions.
        boundary_conditions: int or int[4]
            It can be +1 (periodic), -1 (anti-periodic) or four floats (twisted)
            i.e. a phase proportional to M_PI * [T, Z, Y, X] will multiplies
            the links in the respective directions.
        number_of_levels: int
            Number of levels for the multi-grid, from 1 (no MG) to 4 (max number of levels)
        number_openmp_threads: int
            Number of openmp threads, from 1 to omp_get_num_threads()
        init_file: str
            Input file in DDalphaAMG format
        rnd_seeds: list(int)
            An int per MPI process (list of length comm.size)
        """
        self._init_params = lib.DDalphaAMG_init()
        self._run_params = lib.DDalphaAMG_parameters()
        self._status = lib.DDalphaAMG_status()
        self._setup = -1
        self.updated = True

        global_lattice, block_lattice, procs, comm = get_lattice_partitioning(
            global_lattice, block_lattice, procs, comm
        )

        self._init_params.comm_cart = cast["MPI_Comm"](MPI._handleof(comm))
        self._init_params.Cart_rank = nullptr
        self._init_params.Cart_coords = nullptr

        if boundary_conditions == 1:
            self._init_params.bc = 0
        elif boundary_conditions == -1:
            self._init_params.bc = 1
        else:
            if (
                not isiterable(boundary_conditions, (int, float))
                and len(boundary_conditions) == 4
            ):
                raise TypeError(
                    """
                boundary_conditions can be +1 (periodic), -1 (anti-periodic) or four floats
                (twisted), i.e. a phase proportional to M_PI * [T, Z, Y, X] multiplies links
                in the respective directions.
                """
                )
            self._init_params.bc = 2

        for i in range(4):
            self._init_params.global_lattice[i] = global_lattice[i]
            self._init_params.procs[i] = procs[i]

            self._init_params.block_lattice[i] = block_lattice[i]

            if self._init_params.bc == 2:
                self._init_params.theta[i] = boundary_conditions[i]
            else:
                self._init_params.theta[i] = 0

        self._init_params.number_of_levels = number_of_levels
        self._init_params.number_openmp_threads = number_openmp_threads

        self._init_params.kappa = kwargs.pop("kappa", 0)
        self._init_params.mu = kwargs.pop("mu", 0)
        self._init_params.csw = kwargs.pop("csw", 0)

        if init_file:
            self._init_params.init_file = init_file
        if rnd_seeds:
            if not isiterable(rnd_seeds, int):
                raise TypeError("rnd_seeds needs to be a list of int")
            if not len(rnd_seeds) == comm.size:
                raise ValueError("rnd_seeds must be of length comm.size")
            self._init_params.rnd_seeds = array("I", rnd_seeds)

        if Solver.initialized:
            self.__del__()
            logging.warning(
                """
                The solver library was already initialized on this node.
                The previously initialized Solver class cannot be used anymore!
                NOTE: The DDalphaAMG library supports only one Solver at time.
                """
            )
        lib.DDalphaAMG_initialize(self._init_params, self._run_params, self._status)
        Solver.inizialized = True

        kwargs.setdefault("print", 1)
        kwargs.setdefault("mixed_precision", 2)
        kwargs.setdefault("method", 2)
        kwargs.setdefault(
            "interpolation", 0 if self.nlevels == 1 else self._run_params.interpolation
        )
        if kwargs:
            self.update_parameters(**kwargs)

    def __del__(self):
        if Solver.initialized:
            lib.DDalphaAMG_finalize()
            Solver.inizialized = False

    @property
    def nlevels(self) -> Global:
        "Number of levels of the solver"
        return self.number_of_levels

    @property
    def global_lattice(self) -> Global:
        "Global lattice size of the solver"
        return tuple(self._init_params.global_lattice)

    @property
    def block_lattice(self) -> Global:
        "Block lattice size of the solver"
        return tuple(self._init_params.block_lattice)

    @property
    def procs(self) -> Global:
        "Number of MPI processes per direction"
        return tuple(self._init_params.procs)

    @compute_property
    def local_lattice(self) -> Global:
        "Local lattice size of the solver"
        return tuple(
            i // j for i, j in zip(tuple(self.global_lattice), tuple(self.procs))
        )

    def check_status(self):
        "Checks the current MG status"
        assert self._status.success == 1

    def cast_vector(self, vec):
        "Casts a vector into a format suitable for the solver"
        assert vec.shape == tuple(self.local_lattice) + (
            4,
            3,
        ), f"""
        Given array has not compatible shape.
        array shape = {vec.shape}
        expected shape = {tuple(self.local_lattice) + (4, 3)}
        """
        return numpy.array(vec, dtype="complex128", copy=False)

    def zeros(self) -> Array:
        "Creates a new vector suitable for the solver"
        shape = tuple(self.local_lattice) + (4, 3)
        return numpy.zeros(shape, dtype="complex128")

    def random(self) -> Array:
        "Creates a new random vector using DDalphaAMG function"
        arr = self.zeros()
        lib.DDalphaAMG_define_vector_rand(arr)
        return arr

    def update_parameters(self, **kwargs):
        "Updates multi-grid parameters given in kwargs"
        for key, val in kwargs.items():
            setattr(self, key, val)
        if not self.updated:
            lib.DDalphaAMG_update_parameters(self._run_params, self._status)
            self.updated = True

    @property
    def setup_status(self):
        """
        Number of setup iterations performed.
          If -1, then no configuration has been loaded,
          if  0, then no setup has been done,
          if  n, then n setup iterations have been performed.
        """
        return self._setup

    @property
    def comm(self):
        "Returns the MPI communicator used by the library."
        comm = MPI.Comm()
        comm_ptr = to_pointer(MPI._addressof(comm), "MPI_Comm*")
        lib.MPI_Comm_free(comm_ptr)
        lib.DDalphaAMG_get_communicator(comm_ptr)
        return MPI.Cartcomm(comm)

    @property
    def coords(self):
        "Returns the coordinates of the MPI communicator"
        return tuple(self.comm.Get_topo()[2])

    def setup(self):
        "Runs the setup. If called again, the setup is re-run."
        if self.setup_status < 0:
            raise RuntimeError("No configuration loaded. See set_configuration")
        self.update_parameters()
        lib.DDalphaAMG_setup(self._status)
        self._setup = self._status.success

    def update_setup(self, iterations=1):
        "Runs more setup iterations."
        self.update_parameters()
        lib.DDalphaAMG_update_setup(iterations, self._status)
        self._setup = self._status.success

    @property
    def setup_done(self) -> Global:
        "Returns if setup has been done"
        return self.setup_status > 0

    def solve(self, rhs, tolerance=1e-9) -> Array:
        "Solves D*x=rhs and returns x at the required tolerance."
        rhs = self.cast_vector(rhs)
        sol = self.zeros()

        if not self.setup_done:
            self.setup()
        self.update_parameters()

        lib.DDalphaAMG_solve(sol, rhs, tolerance, self._status)
        self.check_status()
        return sol

    def apply_operator(self, vec) -> Array:
        "Applies the Dirac operator on the given vector"
        vec = self.cast_vector(vec)
        res = self.zeros()

        if not self.setup_done:
            self.setup()
        self.update_parameters()

        lib.DDalphaAMG_apply_operator(res, vec, self._status)
        self.check_status()
        return res

    # alias for apply_operator
    D = apply_operator

    def __dir__(self):
        keys = set(object.__dir__(self))
        for key in dir(self._init_params) + dir(self._run_params):
            if not key.startswith("_"):
                keys.add(key)
        return sorted(keys)

    def __getattr__(self, key):
        if key in Solver.__slots__:
            return object.__getattribute__(self, key)

        try:
            return getattr(self._run_params, key)
        except AttributeError:
            return getattr(self._init_params, key)

    def __setattr__(self, key, val):
        if key in Solver.__slots__:
            object.__setattr__(self, key, val)
            return

        if hasattr(self._run_params, key):
            setattr(self._run_params, key, val)
            self.updated = False
            return

        if hasattr(self._init_params, key):
            raise RuntimeError("An initialization parameter cannot be changed.")

        raise AttributeError(f"{key} is not a runtime parameter")

    def read_configuration(self, filename: abspath, fileformat="lime") -> Array:
        "Reads configuration from file"
        formats = ["DDalphaAMG", "lime"]
        filename = realpath(filename)
        assert fileformat in formats, "fileformat must be one of %s" % formats
        assert isfile(filename), "Filename %s does not exist" % filename
        if fileformat == "lime":
            assert WITH_CLIME, "clime not enabled"

        shape = tuple(self.local_lattice) + (4, 3, 3)
        conf = numpy.zeros(shape, dtype="complex128")
        lib.DDalphaAMG_read_configuration(
            conf, filename, formats.index(fileformat), self._status
        )
        return conf

    def set_configuration(self, conf) -> Global:
        "Sets the configuration to be used in the Dirac operator."
        assert conf.shape == tuple(self.local_lattice) + (
            4,
            3,
            3,
        ), f"""
        Given array has not compatible shape.
        array shape = {conf.shape}
        expected shape = {tuple(self.local_lattice) + (4, 3, 3)}
        """

        conf = numpy.array(conf, dtype="complex128", copy=False)
        lib.DDalphaAMG_set_configuration(conf, self._status)
        self._setup = 0
        return self._status.info


def get_lattice_partitioning(global_lattice, block_lattice=None, procs=None, comm=None):
    """
    Checks or determines the block_lattice and procs based on the given global_lattice and comm.
    """
    if not len(global_lattice) == 4:
        raise ValueError("global_lattice must be a list of length 4 (T, Z, Y, X)")

    local_lattice = list(global_lattice)
    if block_lattice:
        if not len(block_lattice) == 4:
            raise ValueError("block_lattice must be a list of length 4 (T, Z, Y, X)")
        if not all((i % j == 0 for i, j in zip(global_lattice, block_lattice))):
            raise ValueError(
                "block_lattice must divide the global_lattice %s %% %s = 0"
                % (
                    global_lattice,
                    block_lattice,
                )
            )
        local_lattice = [i // j for i, j in zip(local_lattice, block_lattice)]

    if procs:
        if not len(procs) == 4:
            raise ValueError("procs must be a list of length 4 (T, Z, Y, X)")

    if comm is not None:
        if not isinstance(comm, MPI.Comm):
            raise TypeError(f"Expected an MPI.Comm but got {type(comm)}")

        if isinstance(comm, MPI.Cartcomm):
            if not comm.dim == 4:
                raise ValueError("Expected a Cartesian comm of size 4")
            if not comm.periods == [1, 1, 1, 1]:
                raise ValueError(
                    "Expected a Cartesian with periodic boundary condiitons"
                )
            if procs:
                if not list(procs) == comm.dims:
                    raise ValueError(f"procs={procs} and comm.dims={comm.dims} differ")
            else:
                procs = comm.dims
    else:
        comm = default_comm()

    num_workers = comm.size

    if procs:
        if not numpy.prod(procs) == num_workers:
            raise ValueError(
                "The number of workers (%d) does not match the given procs %s"
                % (
                    num_workers,
                    procs,
                )
            )
        if not all((i % j == 0 for i, j in zip(local_lattice, procs))):
            raise ValueError(
                "procs must divide the local_lattice %s %% %s = 0"
                % (local_lattice, procs)
            )
        local_lattice = [i // j for i, j in zip(local_lattice, procs)]
    else:
        procs = [1] * 4
        for factor in reversed(list(prime_factors(num_workers))):
            for local in reversed(sorted(local_lattice)):
                if local % factor == 0:
                    idx = local_lattice.index(local)
                    local_lattice[idx] //= factor
                    procs[idx] *= factor
                    factor = 1
                    break
            if factor != 1:
                raise RuntimeError(
                    """
                Could not create the list of procs:
                num_workers = %d
                factors = %s
                global_lattice = %s
                """
                    % (
                        num_workers,
                        factors,
                        global_lattice,
                    )
                )
        logging.info("Determined procs is %s for %d workers", procs, num_workers)

    if not block_lattice:
        block_lattice = [1] * 4
        # DDalphaAMG requires at least one direction to be multiple of 2
        for local in reversed(sorted(local_lattice)):
            if local % 2 == 0:
                idx = local_lattice.index(local)
                local_lattice[idx] //= 2
                break
        # An optimal block size is 4. Here we find the closest factor to 4
        optimal = 4 if min(global_lattice) > 4 else 2
        for i, local in enumerate(local_lattice):
            get_ratio = lambda i: i / optimal if i >= optimal else optimal / i
            best = get_ratio(1)
            for factor in factors(local):
                if get_ratio(factor) <= best:
                    best = get_ratio(factor)
                    block_lattice[i] = factor
                else:
                    break
        logging.info("Determined block_lattice is %s", block_lattice)

    return global_lattice, block_lattice, procs, comm
