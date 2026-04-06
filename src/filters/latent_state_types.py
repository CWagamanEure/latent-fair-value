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
    spot_error: float | None = None
    perp_error: float | None = None
    temporary_dislocation: float | None = None
    quoted_spot_price: float | None = None
    quoted_perp_price: float | None = None
    quoted_basis: float | None = None
    raw_state_vector_json: str | None = None
    raw_covariance_matrix_json: str | None = None

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
            basis=float(vector[1]),
        )


@dataclass(frozen=True)
class PriceBasisErrorState(BaseLatentState):
    log_spot: float
    log_basis: float
    spot_error: float
    perp_error: float

    @property
    def vector(self) -> Vector:
        return np.array(
            [self.log_spot, self.log_basis, self.spot_error, self.perp_error],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(
            cls,
            timestamp: int,
            vector: Vector,
    ) -> "PriceBasisErrorState":
        if vector.shape != (4,):
            raise ValueError(f"Expected state vector of shape (4,), got {vector.shape}")
        return cls(
            timestamp=timestamp,
            log_spot=float(vector[0]),
            log_basis =float(vector[1]),
            spot_error =float(vector[2]),
            perp_error=float(vector[3]),

        )

    @property
    def equilibrium_log_spot(self) -> float:
        return self.log_spot

    @property
    def equilibrium_log_perp(self) -> float:
        return self.log_spot + self.log_basis

    @property
    def temporary_dislocation(self) -> float:
        return self.perp_error - self.spot_error

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
        if not np.allclose(self.matrix, self.matrix.T, atol=1e-10):
            raise ValueError("Covariance must be symmetric")


@dataclass
class Covariates:
    timestamp: int | None = None
    funding_rate: float | None = None
    open_interest_change: float | None = None
    ofi: float | None = None
    queue_imbalance: float | None = None
    spread: float | None = None
    spot_rv_15m: float | None = None




@dataclass(frozen=True)
class FilterSettings:
    asset: str
    price_choice: str
    # Init
    init_state: PriceBasisErrorState 
    init_cov: StateCovariance

    # price var and basis var dependent on time since last update
    price_var_per_sec: float
    basis_var_per_sec: float

    # if 0 basis acts as random walk
    error_kappa: float
    basis_kappa: float 
    basis_long_run_mean: float

    spot_error_var_per_sec: float
    perp_error_var_per_sec: float


    # measurement error values
    microprice_r_mult: float
    perp_r_mult: float
    spot_r_mult: float 

    min_measurement_var: float

    # covariates
    covariates: Optional[Covariates]

    # hard coded to btc rn
    basis_target_intercept_bps: float = -4.9220
    basis_target_funding_coef: float = 2.463e4
    max_covariate_age: int | None = 60_000
    


    










