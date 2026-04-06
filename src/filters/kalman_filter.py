import numpy as np

from src.exceptions import IncorrectAssetException, StaleMeasurementException
from src.filters.base_filter import BaseFilter
from src.filters.latent_state_types import (
    FilterSettings,
    PriceBasisErrorState,
    StateCovariance,
    Covariates,
)
from src.measurement_types import BBOMeasurement, BaseMeasurement


class KalmanFilter(BaseFilter):
    def __init__(
        self,
        filter_settings: FilterSettings,
    ):
        self.asset = filter_settings.asset.upper()
        self.price_choice = filter_settings.price_choice

        # x[0] = latent log reference/spot price
        # x[1] = latent log basis = log(perp) - log(spot)
        # x[2] = transient spot quote error
        # x[3] = transient perp quote error
        self.state: PriceBasisErrorState = filter_settings.init_state
        self.cov: StateCovariance = filter_settings.init_cov

        state_dim = self.state.vector.shape[0]
        if self.cov.matrix.shape != (state_dim, state_dim):
            raise ValueError(
                f"Initial covariance shape {self.cov.matrix.shape} does not match state dimension {state_dim}"
            )
        if self.cov.timestamp != self.state.timestamp:
            raise ValueError(
                "Initial covariance timestamp does not match initial state timestamp"
            )

        self.microprice_r_mult = filter_settings.microprice_r_mult
        self.perp_r_mult = filter_settings.perp_r_mult
        self.spot_r_mult = filter_settings.spot_r_mult
        self.min_measurement_var = filter_settings.min_measurement_var

        self.price_var_per_sec = filter_settings.price_var_per_sec
        self.basis_var_per_sec = filter_settings.basis_var_per_sec

        self.basis_kappa = filter_settings.basis_kappa
        self.basis_long_run_mean = filter_settings.basis_long_run_mean

        self.spot_error_var_per_sec = filter_settings.spot_error_var_per_sec
        self.perp_error_var_per_sec = filter_settings.perp_error_var_per_sec
        self.error_kappa = filter_settings.error_kappa

        # Covariates
        self.covariates: Covariates = getattr(filter_settings, "covariates", None)
        if self.covariates is None:
            self.covariates = Covariates()

        self.max_covariate_age_ms = getattr(
            filter_settings,
            "max_covariate_age_ms",
            getattr(filter_settings, "max_covariate_age", 60_000),
        )

        # Basis target coefficients
        self.basis_target_intercept_bps = getattr(
            filter_settings, "basis_target_intercept_bps", 0.0
        )
        self.basis_target_funding_coef = getattr(
            filter_settings, "basis_target_funding_coef", 0.0
        )

    def _dt_seconds(self, new_timestamp: int | None) -> float:
        old_timestamp = self.state.timestamp
        if old_timestamp is None or new_timestamp is None:
            return 0.0

        dt_ms = new_timestamp - old_timestamp
        if dt_ms < 0:
            raise StaleMeasurementException(
                f"Measurement timestamp {new_timestamp} is older than filter state timestamp: {old_timestamp}"
            )

        return dt_ms / 1000.0

    def _transition_and_process(self, dt_sec: float):
        # state 0: log price random walk
        q_price = self.price_var_per_sec * dt_sec

        # states 1,2,3: OU processes
        phi_b, q_basis = self._ou_discretization(
            self.basis_var_per_sec, self.basis_kappa, dt_sec
        )
        phi_s, q_spot_err = self._ou_discretization(
            self.spot_error_var_per_sec, self.error_kappa, dt_sec
        )
        phi_p, q_perp_err = self._ou_discretization(
            self.perp_error_var_per_sec, self.error_kappa, dt_sec
        )

        F = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, phi_b, 0.0, 0.0],
                [0.0, 0.0, phi_s, 0.0],
                [0.0, 0.0, 0.0, phi_p],
            ],
            dtype=np.float64,
        )

        theta_log = self._current_basis_target_log()
        c = np.array(
            [
                0.0,
                theta_log * (1.0 - phi_b),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        Q = np.array(
            [
                [max(q_price, 0.0), 0.0, 0.0, 0.0],
                [0.0, max(q_basis, 0.0), 0.0, 0.0],
                [0.0, 0.0, max(q_spot_err, 0.0), 0.0],
                [0.0, 0.0, 0.0, max(q_perp_err, 0.0)],
            ],
            dtype=np.float64,
        )

        return F, Q, c

    def predict(self, timestamp: int | None = None) -> PriceBasisErrorState:
        dt_sec = self._dt_seconds(timestamp)

        F, Q, c = self._transition_and_process(dt_sec)

        x = self.state.vector
        P = self.cov.matrix

        x_pred = F @ x + c
        P_pred = F @ P @ F.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        ts = self.state.timestamp if timestamp is None else timestamp
        self.state = PriceBasisErrorState.from_vector(timestamp=ts, vector=x_pred)
        self.cov = StateCovariance(timestamp=ts, matrix=P_pred)
        return self.state

    def update(self, measurement: BaseMeasurement) -> PriceBasisErrorState:
        self._verify_measurement(measurement)

        if measurement.timestamp is not None:
            self.predict(measurement.timestamp)

        # z = observed measurement
        # H = measurement matrix
        # R = measurement noise covariance
        z, H, R = self._measurement_model(measurement)

        x = self.state.vector
        P = self.cov.matrix

        # Innovation
        y = z - H @ x
        S = H @ P @ H.T + R

        if not np.isfinite(S).all() or S[0, 0] <= 0.0:
            raise ValueError(f"Invalid innovation covariance S: {S}")

        # Scalar-measurement Kalman gain
        K = (P @ H.T) / S[0, 0]

        # Mean update
        x_new = x + (K @ y)

        # Joseph covariance update
        I = np.eye(P.shape[0], dtype=np.float64)
        KH = K @ H
        P_new = (I - KH) @ P @ (I - KH).T + K @ R @ K.T
        P_new = 0.5 * (P_new + P_new.T)

        ts = measurement.timestamp if measurement.timestamp is not None else self.state.timestamp
        self.state = PriceBasisErrorState.from_vector(timestamp=ts, vector=x_new)
        self.cov = StateCovariance(timestamp=ts, matrix=P_new)
        return self.state

    def update_covariates(
        self,
        timestamp: int | None = None,
        funding_rate: float | None = None,
    ) -> None:
        if timestamp is not None:
            self.covariates.timestamp = int(timestamp)
        if funding_rate is not None:
            self.covariates.funding_rate = float(funding_rate)

    def _covariates_are_fresh(self) -> bool:
        cov_ts = self.covariates.timestamp
        state_ts = self.state.timestamp

        if self.max_covariate_age_ms is None:
            return True

        if cov_ts is None or state_ts is None:
            return False

        age_ms = state_ts - cov_ts
        return 0 <= age_ms <= self.max_covariate_age_ms

    def _measurement_model(self, measurement):
        if not isinstance(measurement, BBOMeasurement):
            raise TypeError(f"Unsupported measurement type: {type(measurement).__name__}")

        if measurement.bid_price is None or measurement.ask_price is None:
            raise ValueError("BBOMeasurement is missing bid/ask prices")
        if measurement.bid_size is None or measurement.ask_size is None:
            raise ValueError("BBOMeasurement is missing bid/ask sizes")

        spread = measurement.ask_price - measurement.bid_price
        half_spread = max(0.5 * float(spread), 1e-8)
        depth = max(float(measurement.bid_size + measurement.ask_size), 1e-8)

        # Simple heuristic measurement variance in price space
        base_var_price = half_spread**2 / np.sqrt(depth)

        if self.price_choice == "microprice":
            price_value = measurement.microprice()
            if price_value is None:
                raise ValueError("BBOMeasurement is missing data required to compute microprice")
            base_var_price *= self.microprice_r_mult
        elif self.price_choice == "midprice":
            price_value = measurement.mid()
            if price_value is None:
                raise ValueError("BBOMeasurement is missing info to compute the midprice")
        else:
            raise ValueError(f"Unknown price choice: {self.price_choice}")

        price_value = float(price_value)
        if price_value <= 0.0:
            raise ValueError(f"Price must be positive to take the log, got {price_value}")

        if measurement.is_perp:
            base_var_price *= self.perp_r_mult
            H = np.array([[1.0, 1.0, 0.0, 1.0]], dtype=np.float64)
        elif measurement.is_spot:
            base_var_price *= self.spot_r_mult
            H = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float64)
        else:
            raise ValueError("BBOMeasurement must be either spot or perp")

        z = np.array([np.log(price_value)], dtype=np.float64)

        # Delta-method conversion from price variance to log-price variance
        base_var_log = base_var_price / max(price_value**2, 1e-12)
        R = np.array([[max(self.min_measurement_var, base_var_log)]], dtype=np.float64)

        return z, H, R

    def _verify_measurement(self, measurement):
        if measurement.asset.upper() != self.asset:
            raise IncorrectAssetException(
                f"{measurement.asset} does not match filter asset of {self.asset}"
            )

    def _ou_discretization(self, var_per_sec: float, kappa: float, dt_sec: float):
        if kappa <= 0.0:
            return 1.0, max(var_per_sec * dt_sec, 0.0)

        phi = np.exp(-kappa * dt_sec)
        q = var_per_sec * (1.0 - np.exp(-2.0 * kappa * dt_sec)) / (2.0 * kappa)
        return phi, max(q, 0.0)

    def _current_basis_target_log(self) -> float:
        if not self._covariates_are_fresh():
            return float(self.basis_long_run_mean)

        funding = (
            0.0
            if self.covariates.funding_rate is None
            else float(self.covariates.funding_rate)
        )

        theta_bps = (
            self.basis_target_intercept_bps
            + self.basis_target_funding_coef * funding
        )

        theta_bps = float(np.clip(theta_bps, -200.0, 200.0))
        theta_log = np.log(np.clip(1.0 + theta_bps / 10000.0, 1e-8, None))
        return float(theta_log)

    @property
    def equilibrium_log_spot(self) -> float:
        return float(self.state.vector[0])

    @property
    def equilibrium_log_perp(self) -> float:
        return float(self.state.vector[0] + self.state.vector[1])

    @property
    def transient_spot_error(self) -> float:
        return float(self.state.vector[2])

    @property
    def transient_perp_error(self) -> float:
        return float(self.state.vector[3])

    @property
    def temporary_dislocation(self) -> float:
        return float(self.state.vector[3] - self.state.vector[2])

    @property
    def equilibrium_spot_price(self) -> float:
        return float(np.exp(self.state.log_spot))

    @property
    def equilibrium_perp_price(self) -> float:
        return float(np.exp(self.state.log_spot + self.state.log_basis))

    @property
    def quoted_spot_price(self) -> float:
        return float(np.exp(self.state.log_spot + self.state.spot_error))

    @property
    def quoted_perp_price(self) -> float:
        return float(np.exp(self.state.log_spot + self.state.log_basis + self.state.perp_error))

    @property
    def equilibrium_dollar_basis(self) -> float:
        return self.equilibrium_perp_price - self.equilibrium_spot_price

    @property
    def quoted_dollar_basis(self) -> float:
        return self.quoted_perp_price - self.quoted_spot_price

    @property
    def spot_error_dollars(self) -> float:
        return self.quoted_spot_price - self.equilibrium_spot_price

    @property
    def perp_error_dollars(self) -> float:
        return self.quoted_perp_price - self.equilibrium_perp_price

    @property
    def temporary_dislocation_dollars(self) -> float:
        return self.quoted_dollar_basis - self.equilibrium_dollar_basis
