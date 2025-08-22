from dataclasses import dataclass
import numpy as np
from typing import Any, Optional, Literal

@dataclass(frozen=True)
class GeometryConfig:

    dims : int                              # number of dimensions
    N : int                                 # number of QDs in each dimension
    nc_edgelength : float = 8               # length of each QD (units ?)
    ligand_length : float = 1               # length of each ligand (units ?)

    @property
    def qd_spacing(self) -> float:
        return self.nc_edgelength + 2 * self.ligand_length

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
    J_c: float
    relative_spatial_disorder: float = 0.0      # relative spatial disorder
    dipolegen: Any = "random"                   # dipole generation procedure
    seed_base: int = 12345                      # root seed for disorder draws (realization-level)
                                                # can be drawn randomly to make experiments random
                                                # keep default for reproducibility


@dataclass(frozen=True)
class BathConfig:

    temp: float

    w_c: float                                  # cutoff frequency for bath (units ?)
    reorg_nrg: float                            # reorganization energy (units ?) 
    spectral_density: Any = "cubic-exp"

    @property
    def spectrum(self) -> list:
        return [self.spectral_density, self.reorg_nrg, self.w_c]


RatesBy = Literal["radius", "weight"]

@dataclass(frozen=False)
class RunConfig:

    ntrajs: int
    nrealizations: int
    t_final: float = 5                      # time for each trajectory (units ?)
    time_grid_density: int = 200            # points per unit time for MSD time grid
    max_workers: Optional[int] = None       # max_workers to conduct parallel work

    # mode selector to compute rates in KMC 
    # the selection here determines the simplification scheme for the Redfield rates
    rates_by: RatesBy = "radius"

    # --- parameters for selection by radius ...
    r_hop: Optional[float] = None
    r_ove: Optional[float] = None

    # --- parameters for selection by weight ...
    theta_site: Optional[float] = None
    theta_pol: Optional[float] = None


@dataclass(frozen=True)
class ConvergenceTuneConfig:
    """
    immutable configuration for convergence tuning of KMC hyperparameters parameters(θ_sites, θ_pol).
    bundles algorithm parameters and execution controls.
    """
    # --- site cutoff tuning
    theta_site_lo: float = 0.10
    theta_site_hi: float = 0.02

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

