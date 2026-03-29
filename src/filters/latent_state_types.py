from typing import Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class BaseLatentState:
    timestamp: int



@dataclass(frozen=True)
class PriceBasisState(BaseLatentState):
    price: float
    basis: float

    @property
    def vector(self) -> Vector: 
        return np.array([self.price, self.basis], dtype=np.float64)

    @classmethod
    def from_vector(
            cls,
            timestamp: int,
            vector: Vector, 
    ) -> "PriceBasisState":
        return cls(
            timestamp=timestamp,
            price=float(vector[0]),
            basis=float(vector[1])
        )


@dataclass(frozen=True)
class PriceBasisDisState(PriceBasisState):
    temporary_dislocation: float

@dataclass(frozen=True)
class StateCovariance:
    timestamp: int
    matrix: Matrix

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("Covariance must be 2D")
        rows, cols = self.matrix.shape
        if rows != cols:
            raise ValueError("Covariance must be square")






@dataclass(frozen=True)
class Covariates:
    funding_rate: float
    open_interest_change: float
    ofi: float
    queue_imbalance: float
    spread: float




@dataclass(frozen=True)
class FilterSettings:
    asset: str
    price_choice: str
    # Init
    init_state: PriceBasisState 
    init_cov: StateCovariance

    process_noise: Matrix
    transition_matrix: Matrix


    # measurement error values
    microprice_r_mult: float
    perp_r_mult: float
    spot_r_mult: float 

    min_measurement_var: float

    # covariates
    covariates: Optional[Covariates]
    


    














