# src/qdotkmc/__init__.py
from . import const, lattice, hamiltonian, montecarlo, redfield, utils, plot_utils, convergence

__all__ = ["const", "lattice", "hamiltonian", "montecarlo",
           "redfield", "utils", "plot_utils", "convergence"]

__version__ = "1.0.2"


"""
Version: 1.0.2 
Date: 2025-08-22

Notes:
-   contains weight-based strategy to compute Redfield rates in a computationally efficient way
-   contains convergence module to automatically enable tuning of hyperparamters θ_pol and θ_site
    to optimal values before running KMC calculations
"""