"""Heston calibration: parameter recovery on a synthetic surface, batched COS == scalar COS, and the
vectorised implied-vol inverter against Brent."""

import numpy as np
import pytest

from pricers import bs, heston
from pricers.calibrate import Surface, calibrate

TRUE = heston.HestonParams(v0=0.04, kappa=1.5, theta=0.05, sigma_v=0.6, rho=-0.7)
STRIKES = [70, 80, 90, 95, 100, 105, 110, 120, 130]
MATS = [0.1, 0.25, 0.5, 1.0, 2.0]


@pytest.fixture(scope="module")
def surface():
    return Surface.from_params(100.0, 0.02, 0.01, STRIKES, MATS, TRUE)


def test_price_strikes_equals_scalar_price_to_machine_precision():
    Ks = np.array([60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 150.0])
    for T in (0.05, 0.5, 2.0):
        for right in ("C", "P"):
            batch = heston.price_strikes(100.0, Ks, T, right, TRUE, 0.02, 0.01)
            single = np.array([heston.price(100.0, k, T, right, TRUE, 0.02, 0.01) for k in Ks])
            assert np.max(np.abs(batch - single)) < 1e-12


def test_vectorised_implied_vol_matches_brent_and_the_generating_vol():
    rng = np.random.default_rng(1)
    S, K, T, sig = 100.0, rng.uniform(50, 160, 600), rng.uniform(0.02, 3, 600), rng.uniform(0.05, 1.2, 600)
    for right in ("C", "P"):
        px = np.array([bs.price(S, k, t, s, right, 0.03, 0.01) for k, t, s in zip(K, T, sig, strict=True)])
        v = bs.implied_vol_vec(px, S, K, T, right, 0.03, 0.01)
        ok = np.isfinite(v)
        assert ok.mean() > 0.95                                     # only sub-1e-10·S prices are declared uninformative
        assert np.max(np.abs(v[ok] - sig[ok])) < 1e-7
        brent = np.array([bs.implied_vol(p, S, k, t, right, 0.03, 0.01) for p, k, t in zip(px[ok][:100], K[ok][:100], T[ok][:100], strict=True)])
        assert np.max(np.abs(v[ok][:100] - brent)) < 1e-8
    assert np.all(np.isnan(bs.implied_vol_vec([-1.0, 1e9, 1e-14], S, 100.0, 1.0, "C")))


def test_calibration_recovers_the_generating_parameters(surface):
    res = calibrate(surface)
    p = res.params
    assert abs(p.v0 - TRUE.v0) < 1e-5 and abs(p.rho - TRUE.rho) < 1e-4
    assert abs(p.kappa - TRUE.kappa) < 1e-3 and abs(p.theta - TRUE.theta) < 1e-5 and abs(p.sigma_v - TRUE.sigma_v) < 1e-3
    assert res.rmse_vol < 1e-6 and res.max_abs_vol < 1e-5 and res.n_points == 45
    assert not res.feller and all(t["rmse_vol"] < 1e-5 for t in res.starts)    # every start converged to the same surface


def test_noisy_surface_fits_within_the_noise_and_reports_it(surface):
    rng = np.random.default_rng(0)
    noisy = Surface(surface.S, surface.r, surface.q, surface.K, surface.T, surface.iv + rng.normal(0, 0.002, len(surface.iv)))
    res = calibrate(noisy, starts=[heston.HestonParams(0.09, 1.0, 0.06, 0.8, -0.3)])
    assert 0.001 < res.rmse_vol < 0.003                                # about the 20 bp noise, not below it
    assert abs(res.params.rho - TRUE.rho) < 0.05 and abs(res.params.v0 - TRUE.v0) < 0.005
    assert "RMSE" in res.summary() and "Feller" in res.summary()


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        calibrate(Surface(100.0, 0.0, 0.0, np.array([100.0, 110.0]), np.array([1.0, 1.0]), np.array([0.2, 0.21])))
