import numpy as np

from src.exceptions import IncorrectAssetException
from src.filters.base_filter import BaseFilter
from src.filters.latent_state_types import FilterSettings, PriceBasisState, StateCovariance
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
        self.state: PriceBasisState = filter_settings.init_state
        self.cov: StateCovariance = filter_settings.init_cov 

        self.microprice_r_mult = filter_settings.microprice_r_mult
        self.perp_r_mult = filter_settings.perp_r_mult
        self.spot_r_mult = filter_settings.spot_r_mult
        self.min_measurement_var = filter_settings.min_measurement_var

        self.price_var_per_sec = filter_settings.price_var_per_sec
        self.basis_var_per_sec = filter_settings.basis_var_per_sec

        self.basis_kappa = filter_settings.basis_kappa
        self.basis_long_run_mean = filter_settings.basis_long_run_mean

    def _dt_seconds(self, new_timestamp: int | None) -> float:
        old_timestamp = self.state.timestamp
        if old_timestamp is None or new_timestamp is None:
            return 0.0
        
        dt_ms = new_timestamp - old_timestamp
        if dt_ms < 0:
            raise ValueError(f"Measurement timestamp {new_timestamp} is older than filter state timestamp: {old_timestamp}")

        return dt_ms / 1000.0
        
    def _transition_and_process(self, dt_sec: float):
        # state 0: log price random walk
        q_price = self.price_var_per_sec * dt_sec 

        # State 1: log basis
        # if kappa == 0: random walk
        # Else: OU / mean reversion toward basis_long_run_mean
        if self.basis_kappa <= 0.0:
            phi = 1.0
            q_basis = self.basis_var_per_sec * dt_sec
            c = np.array([0.0, 0.0], dtype=np.float64)
        else:
            # weighted exponentially for time
            # fraction of the past basis that survives to impact this one
            phi = np.exp(-self.basis_kappa * dt_sec)

            q_basis = self.basis_var_per_sec * (1.0 - np.exp(-2.0 * self.basis_kappa * dt_sec)) / (2.0 * self.basis_kappa)

            # c pulls the basis towards the long run mean
            c = np.array([0.0, self.basis_long_run_mean * (1.0 - phi)], dtype=np.float64)

        F = np.array(
            [
                [1.0, 0.0],
                [0.0, phi],
            ],
            dtype=np.float64,
        )

        Q = np.array(
            [
                [max(q_price, 0.0), 0.0],
                [0.0, max(q_basis, 0.0)],
            ],
            dtype=np.float64,
        )
        return F, Q, c


    def predict(self, timestamp: int | None = None) -> PriceBasisState:
        dt_sec = self._dt_seconds(timestamp)

        F, Q, c = self._transition_and_process(dt_sec)

        x = self.state.vector
        P = self.cov.matrix

        x_pred = F @ x + c
        P_pred = F @ P @ F.T + Q

        ts = self.state.timestamp if timestamp is None else timestamp 
        self.state = PriceBasisState.from_vector(timestamp=ts, vector=x_pred)
        self.cov = StateCovariance(timestamp=ts, matrix=P_pred)
        return self.state



    def update(self, measurement: BaseMeasurement) -> PriceBasisState:
        self._verify_measurement(measurement)

        if measurement.timestamp is not None:
            self.predict(measurement.timestamp)

        # z = observed measurement, ex. z=[100.8] when perp mid is 100.8
        # H = measurement matrix, ex. [[1,0]]  when spot observes only the price (not other state)
        # R = Measurement noise covariance
        z, H, R = self._measurement_model(measurement)

        x = self.state.vector
        P = self.cov.matrix

        # y = innovation
        y = z - H @ x
        # S = uncertainty of the residual (covariance of the innovation)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        # new mean
        x_new = x + K @ y
        I = np.eye(P.shape[0], dtype=np.float64)
        KH = K @ H
        P_new = (I - KH) @ P @ (I - KH).T + K @ R @ K.T 

        ts = measurement.timestamp if measurement.timestamp is not None else self.state.timestamp
        self.state = PriceBasisState.from_vector(timestamp=ts, vector=x_new)
        self.cov = StateCovariance(timestamp=ts, matrix=P_new)
        return self.state


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
        # Heuristic (probably fine for now)
        base_var_price = half_spread**2 / np.sqrt(depth)


        if self.price_choice == "microprice":
            price_value = measurement.microprice()
            if price_value is None:
                raise ValueError("BBOMeasurement is missing data required to compute microprice")
            # here we add an uncertainty multiplyer if microprice
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
            H = np.array([[1.0, 1.0]], dtype=np.float64)
        elif measurement.is_spot:
            base_var_price *= self.spot_r_mult
            H = np.array([[1.0, 0.0]], dtype=np.float64)
        else:
            raise ValueError("BBOMeasurement must be either spot or perp")

        z = np.array([np.log(price_value)], dtype=np.float64)
        base_var_log = base_var_price / max(price_value**2, 1e-12)
        R = np.array([[max(self.min_measurement_var, base_var_log)]], dtype=np.float64)
        return z, H, R

    


    def _verify_measurement(self, measurement):
        if measurement.asset.upper() != self.asset:
            raise IncorrectAssetException(f"{measurement.asset} does not match filter asset of {self.asset}") 

