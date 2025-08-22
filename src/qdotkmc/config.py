from dataclasses import dataclass
import numpy as np
from typing import Any, Optional

@dataclass(frozen=True)
class GeometryConfig:

    dims : int
    N : int
    qd_spacing: float

    @property
    def n_sites(self) -> int:
        return self.N ** self.dims

    @property
    def boundary(self) -> float:
        return self.N * self.qd_spacing

    @property
    def lattice_dimension(self) -> np.ndarray:
        return np.array([self.N] * self.dims, dtype=float) * self.qd_spacing


@dataclass(frozen=True)
class DisorderConfig:

    nrg_center: float
    inhomog_sd: float
    relative_spatial_disorder: float
    dipolegen: Any   
    J_c: float
    seed_base: int                      # root seed for disorder draws (realization-level)


@dataclass(frozen=True)
class BathConfig:

    temp: float
    spectrum: Any     # your existing spectrum information object/callable


@dataclass(frozen=False)
class RunConfig:

    ntrajs: int
    nrealizations: int
    t_final: float
    time_grid_density: int = 100            # points per unit time for MSD time grid (NOTE : might want to modify this later)
    max_workers: Optional[int] = None       # max_workers to conduct parallel work

    # --- parameters for selection by radius ...
    r_hop: Optional[float] = None
    r_ove: Optional[float] = None

    # --- parameters for selection by overlap ...
    theta_site: Optional[float] = None
    theta_pol: Optional[float] = None


@dataclass(frozen=True)
class ConvergenceTuneConfig:
    """
    immutable configuration for convergence tuning of KMC hyperparameters parameters(θ_sites, θ_pol).
    bundles algorithm parameters and execution controls.
    """
    # --- site cutoff tuning
    theta_sites_lo: float = 0.10
    theta_sites_hi: float = 0.02

    # --- polaron cutoff tuning
    theta_pol_start: float = 0.30
    theta_pol_min:   float = 0.02

    # --- algorithm hyperparameters
    rho: float       = 0.7      # shrinkage span per iteration
    delta: float     = 0.015    # flatness tolerance (per-octave gain target)
    max_outer: int   = 12       # max steps for θ_sites bisection
    max_steps_pol: int = 8      # max steps for θ_pol shrinkage

    # --- execution control
    no_samples: int = 30
    max_workers: Optional[int] = None       # max_workers to conduct parallel work
    criterion: str  = "rate-displacement"
    verbose: bool   = True

