import numpy as np
import pytest

from src.filters.kalman_filter import KalmanFilter
from src.filters.latent_state_types import FilterSettings, PriceBasisState, StateCovariance


def test_predict_applies_mean_reversion_drift_to_basis_state() -> None:
    initial_state = PriceBasisState(timestamp=1_000, price=4.5, basis=0.4)
    initial_cov = StateCovariance(
        timestamp=1_000,
        matrix=np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64),
    )
    filter_settings = FilterSettings(
        asset="BTC",
        price_choice="midprice",
        init_state=initial_state,
        init_cov=initial_cov,
        price_var_per_sec=0.1,
        basis_var_per_sec=0.2,
        basis_kappa=0.5,
        basis_long_run_mean=0.1,
        microprice_r_mult=1.0,
        perp_r_mult=1.25,
        spot_r_mult=1.0,
        min_measurement_var=1e-8,
        covariates=None,
    )

    kalman_filter = KalmanFilter(filter_settings)
    predicted_state = kalman_filter.predict(timestamp=3_000)

    dt_sec = 2.0
    phi = np.exp(-filter_settings.basis_kappa * dt_sec)
    expected_basis = phi * initial_state.basis + filter_settings.basis_long_run_mean * (1.0 - phi)

    assert predicted_state.timestamp == 3_000
    assert predicted_state.price == pytest.approx(initial_state.price)
    assert predicted_state.basis == pytest.approx(expected_basis)
