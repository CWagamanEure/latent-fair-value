import numpy as np
import pytest

from src.filters.kalman_filter import KalmanFilter
from src.filters.latent_state_types import (
    Covariates,
    FilterSettings,
    PriceBasisErrorState,
    StateCovariance,
)


def test_predict_applies_mean_reversion_drift_to_basis_state() -> None:
    initial_state = PriceBasisErrorState(
        timestamp=1_000,
        log_spot=4.5,
        log_basis=0.4,
        spot_error=0.2,
        perp_error=-0.1,
    )
    initial_cov = StateCovariance(
        timestamp=1_000,
        matrix=np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float64),
    )
    filter_settings = FilterSettings(
        asset="BTC",
        price_choice="midprice",
        init_state=initial_state,
        init_cov=initial_cov,
        price_var_per_sec=0.1,
        basis_var_per_sec=0.2,
        error_kappa=0.75,
        basis_kappa=0.5,
        basis_long_run_mean=0.1,
        spot_error_var_per_sec=0.3,
        perp_error_var_per_sec=0.4,
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
    expected_basis = phi * initial_state.log_basis + filter_settings.basis_long_run_mean * (1.0 - phi)
    error_phi = np.exp(-filter_settings.error_kappa * dt_sec)

    assert predicted_state.timestamp == 3_000
    assert predicted_state.log_spot == pytest.approx(initial_state.log_spot)
    assert predicted_state.log_basis == pytest.approx(expected_basis)
    assert predicted_state.spot_error == pytest.approx(error_phi * initial_state.spot_error)
    assert predicted_state.perp_error == pytest.approx(error_phi * initial_state.perp_error)


def test_init_rejects_covariance_shape_mismatch() -> None:
    initial_state = PriceBasisErrorState(
        timestamp=1_000,
        log_spot=4.5,
        log_basis=0.4,
        spot_error=0.0,
        perp_error=0.0,
    )
    initial_cov = StateCovariance(
        timestamp=1_000,
        matrix=np.eye(2, dtype=np.float64),
    )
    filter_settings = FilterSettings(
        asset="BTC",
        price_choice="midprice",
        init_state=initial_state,
        init_cov=initial_cov,
        price_var_per_sec=0.1,
        basis_var_per_sec=0.2,
        error_kappa=0.75,
        basis_kappa=0.5,
        basis_long_run_mean=0.1,
        spot_error_var_per_sec=0.3,
        perp_error_var_per_sec=0.4,
        microprice_r_mult=1.0,
        perp_r_mult=1.25,
        spot_r_mult=1.0,
        min_measurement_var=1e-8,
        covariates=None,
    )

    with pytest.raises(ValueError, match="does not match state dimension"):
        KalmanFilter(filter_settings)


def test_covariates_are_not_fresh_when_they_are_from_the_future() -> None:
    filter_settings = FilterSettings(
        asset="BTC",
        price_choice="midprice",
        init_state=PriceBasisErrorState(
            timestamp=1_000,
            log_spot=4.5,
            log_basis=0.4,
            spot_error=0.0,
            perp_error=0.0,
        ),
        init_cov=StateCovariance(
            timestamp=1_000,
            matrix=np.eye(4, dtype=np.float64),
        ),
        price_var_per_sec=0.1,
        basis_var_per_sec=0.2,
        error_kappa=0.75,
        basis_kappa=0.5,
        basis_long_run_mean=0.1,
        spot_error_var_per_sec=0.3,
        perp_error_var_per_sec=0.4,
        microprice_r_mult=1.0,
        perp_r_mult=1.25,
        spot_r_mult=1.0,
        min_measurement_var=1e-8,
        covariates=Covariates(timestamp=1_500, funding_rate=0.0001),
    )

    kalman_filter = KalmanFilter(filter_settings)

    assert kalman_filter._covariates_are_fresh() is False
