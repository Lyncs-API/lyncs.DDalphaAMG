from lyncs_setuptools import setup, CMakeExtension

flags = []

try:
    from lyncs_clime import PATHS as lime_path

    flags.append("-DLIME_PATH=%s" % lime_path[0])
except ModuleNotFoundError:
    pass


setup(
    "lyncs_DDalphaAMG",
    exclude=["*.config"],
    ext_modules=[CMakeExtension("lyncs_DDalphaAMG.lib", ".", flags)],
    data_files=[(".", ["config.py.in"])],
    install_requires=[
        "lyncs-mpi",
        "lyncs-cppyy",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"],
    },
    keywords=[
        "Lyncs",
        "DDalphaAMG",
        "Lattice QCD",
        "Multigrid",
        "Wilson",
        "Twisted-mass",
        "Clover",
        "Fermions",
    ],
)
