from dataclasses import dataclass
import numpy as np
from typing import Any

@dataclass(frozen=True)
class GeometryConfig:

    dims : int
    N : int
    qd_spacing: float
    # TODO : eventually move r_hop, r_ove to RunConfig
    r_hop: float
    r_ove: float

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


@dataclass(frozen=True)
class RunConfig:

    ntrajs: int
    nrealizations: int
    t_final: float
    time_grid_density: int = 100  # points per unit time for MSD time grid (NOTE : might want to modify this later)