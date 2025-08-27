# src/qdotkmc/__init__.py
from . import const, lattice, hamiltonian_box, montecarlo, redfield_box, utils, plot_utils

__all__ = ["const", "lattice", "hamiltonian_box", "montecarlo",
           "redfield_box", "utils", "plot_utils"]

__version__ = "1.0.1"


"""
Version: 1.0.1 (Legacy Baseline)
Date: 2025-08-15

Notes:
- This is a baseline version of the code with some good computational power in evaluating the Redfield rates
- Implemented r_hop as hopping radius from center polaron to destination polarons that are r_hop away. 
  We similarly sum over all site in computing the rates that are r_ove away from center polaron.
- Known issues: scaling of diffusivities as r_hop and r_ove increase. There does NOT seem to be convergence
  of rates with those quantities. This makes it essentially impossible to determine convergence
  unless we make r_hop/r_ove almost the same size as the entire lattice, which seems nonsensical because
  we could essentially compute the full Redfield tensor.
"""