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
        self.state: PriceBasisState = filter_settings.init_state
        self.cov: StateCovariance = filter_settings.init_cov 
        self.Q = filter_settings.process_noise
        self.F = filter_settings.transition_matrix

        self.microprice_r_mult = filter_settings.microprice_r_mult
        self.perp_r_mult = filter_settings.perp_r_mult
        self.spot_r_mult = filter_settings.spot_r_mult
        # adding this later 
        self.min_measurement_var = filter_settings.min_measurement_var


    def predict(self) -> PriceBasisState:
        x = self.state.vector
        P = self.cov.matrix

        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q

        ts = self.state.timestamp
        self.state = PriceBasisState.from_vector(timestamp=ts, vector=x_pred)
        self.cov = StateCovariance(timestamp=ts, matrix=P_pred)
        return self.state

    def update(self, measurement: BaseMeasurement) -> PriceBasisState:
        self._verify_measurement(measurement)

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
        I = np.eye(P.shape[0])
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
        base_var = half_spread**2 / np.sqrt(depth)
        if self.price_choice == "microprice":
            price_value = measurement.microprice()
            if price_value is None:
                raise ValueError("BBOMeasurement is missing data required to compute microprice")
            # here we add an uncertainty multiplyer if microprice
            base_var *= self.microprice_r_mult 
        elif self.price_choice == "midprice":
            price_value = measurement.mid()
            if price_value is None:
                raise ValueError("BBOMeasurement is missing info to compute the midprice")
        else:
            raise ValueError(f"Unknown price choice: {self.price_choice}")



        if measurement.is_perp:
            base_var *= self.perp_r_mult 
            H = np.array([[1.0, 1.0]], dtype=np.float64)
        elif measurement.is_spot:
            base_var *= self.spot_r_mult
            H = np.array([[1.0, 0.0]], dtype=np.float64)
        else:
            raise ValueError("BBOMeasurement must be either spot or perp")

        price = np.array([float(price_value)], dtype=np.float64)
        R = np.array([[max(self.min_measurement_var, base_var)]], dtype=np.float64)
        return price, H, R

    


    def _verify_measurement(self, measurement):
        if measurement.asset.upper() != self.asset:
            raise IncorrectAssetException(f"{measurement.asset} does not match filter asset of {self.asset}") 


