# src/qdotkmc/__init__.py
from . import const, lattice, hamiltonian, montecarlo, redfield, utils, plot_utils, convergence, print_utils

__all__ = ["const", "lattice", "hamiltonian", "montecarlo",
           "redfield", "utils", "plot_utils", "convergence",
           "backend", "print_utils"]

__version__ = "1.0.4"


"""
Version: 1.0.4 
Date: 2025-11-20

Notes:
-   contains new spectral densities ('ohmic-exp', 'drude-lorentz')
-   contains 1D, 2D, 3D cubic QD lattices, i.e. 3D finally implemented
-   contains weight-based strategy to compute Redfield rates in a computationally efficient way
-   contains convergence module to automatically enable tuning of hyperparamters θ_pol and θ_site
    to optimal values before running KMC calculations
-   contains automatic GPU/CPU switch and allows running both convergence algorithms and rate computations
    on GPU resources.
-   ...
"""