""" Loading DDalphaAMG via cppyy """

__all__ = [
    "lib",
    "PATHS",
]

from lyncs_mpi import lib as libmpi
from lyncs_cppyy import Lib
from . import __path__
from .config import WITH_CLIME

libraries = [
    "libDDalphaAMG.so",
    libmpi,
]

if WITH_CLIME:
    from lyncs_clime import lib as libclime

    libraries.append(libclime)

PATHS = list(__path__)

lib = Lib(
    path=PATHS,
    header="DDalphaAMG.h",
    library=libraries,
    c_include=True,
    check="DDalphaAMG_init",
)
