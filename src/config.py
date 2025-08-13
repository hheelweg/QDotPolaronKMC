from dataclasses import dataclass
import numpy as np
from typing import Any

@dataclass(frozen=True)
class GeometryConfig:
    dims : int
    sidelength : int
    qd_spacing: float
    r_hop: float
    r_ove: float

    @property
    def n_sites(self) -> int:
        return self.sidelength ** self.dims

    @property
    def boundary(self) -> float:
        return self.sidelength * self.qd_spacing

    @property
    def lattice_dimension(self) -> np.ndarray:
        return np.array([self.sidelength] * self.dims, dtype=float) * self.qd_spacing


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