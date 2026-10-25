"""Metrics recover known answers on synthetic data."""

import numpy
import pytest

from lejepa_norm.eval import downstream
from lejepa_norm.eval import features
from lejepa_norm.eval import metrics
from lejepa_norm.eval import redundancy


def _rng(seed: int = 0) -> numpy.random.Generator:
    return numpy.random.default_rng(seed)


def test_kappa_is_one_for_gaussian_and_zero_for_sphere() -> None:
    d = 256
    gauss = _rng().standard_normal((20000, d))
    assert metrics.kappa(gauss) == pytest.approx(1.0, abs=0.05)
    sphere = features.direction(gauss) * numpy.sqrt(d)
    assert metrics.kappa(sphere) == pytest.approx(0.0, abs=1e-6)


def test_icc_recovers_known_variance_ratio_with_ci() -> None:
    rng = _rng(1)
    n, k = 3000, 8
    sigma_b2, sigma_w2 = 2.0, 1.0  # true ICC = 2 / 3
    y = rng.normal(0, numpy.sqrt(sigma_b2), (n, 1)) + rng.normal(
        0, numpy.sqrt(sigma_w2), (n, k)
    )
    result = metrics.icc1(y, num_bootstrap=200, rng=rng)
    assert result.icc.value == pytest.approx(2 / 3, abs=0.03)
    assert result.icc.low < 2 / 3 < result.icc.high
    assert result.msw == pytest.approx(sigma_w2, rel=0.05)

    pure_noise = rng.normal(size=(n, k))
    assert metrics.icc1(pure_noise).icc.value == pytest.approx(0.0, abs=0.03)


def _sphere(n: int, d: int, seed: int) -> numpy.ndarray:
    return features.direction(_rng(seed).standard_normal((n, d)))


def test_redundancy_r2_near_zero_when_radius_independent_of_direction() -> None:
    u = _sphere(600, 8, 2)
    y = _rng(3).normal(size=600)
    for kernel in ("exp_dot", "linear"):
        fit = redundancy.kernel_ridge_cv(u, y, kernel, 5, _rng(4))
        assert abs(fit.r2) < 0.05, kernel
        numpy.testing.assert_allclose(fit.residuals, y - fit.predictions)


def test_redundancy_r2_high_when_radius_is_smooth_function_of_direction() -> None:
    u = _sphere(600, 8, 5)
    # Smooth but non-linear in u: the kernel model should beat linear ridge.
    y = numpy.sin(3 * u[:, 0]) + u[:, 1] * u[:, 2] * 4
    y += 0.05 * _rng(6).normal(size=y.size)
    kernel_fit = redundancy.kernel_ridge_cv(u, y, "exp_dot", 5, _rng(7))
    linear_fit = redundancy.kernel_ridge_cv(u, y, "linear", 5, _rng(7))
    assert kernel_fit.r2 > 0.8
    assert kernel_fit.r2 > linear_fit.r2 + 0.2


def test_nested_kernel_ridge_reuses_spectra_across_targets() -> None:
    u = _sphere(300, 8, 11)
    rng = _rng(12)
    y1, y2 = rng.normal(size=300), numpy.sin(3 * u[:, 0])
    model = redundancy.NestedKernelRidge(u, "exp_dot", 5, _rng(13))
    for y in (y1, y2, y1):
        expected = redundancy.kernel_ridge_cv(u, y, "exp_dot", 5, _rng(13))
        got = model.fit(y)
        assert got.r2 == pytest.approx(expected.r2, abs=1e-12)
        numpy.testing.assert_allclose(got.predictions, expected.predictions)
        assert got.params == expected.params
    with pytest.raises(ValueError):
        model.fit(y1[:-1])


def test_batched_r2_matches_one_by_one_fits() -> None:
    u = _sphere(300, 8, 11)
    rng = _rng(12)
    ys = numpy.stack([rng.normal(size=300), numpy.sin(3 * u[:, 0])] * 3)
    ys[2:] += 0.3 * rng.normal(size=(4, 300))
    model = redundancy.NestedKernelRidge(u, "exp_dot", 5, _rng(13))
    expected = numpy.array([model.fit(y).r2 for y in ys])
    numpy.testing.assert_allclose(model.r2_many(ys, chunk=4), expected, atol=1e-10)
    assert model.r2_many(ys[:0]).shape == (0,)
    with pytest.raises(ValueError):
        model.r2_many(ys[:, :-1])


def test_permutation_test_separates_dependent_from_independent_radius() -> None:
    u = _sphere(400, 6, 14)
    model = redundancy.NestedKernelRidge(u, "exp_dot", 5, _rng(15))
    dependent = numpy.sin(3 * u[:, 0]) + 0.1 * _rng(16).normal(size=400)
    test = redundancy.permutation_test(model, dependent, 30, _rng(17))
    assert test.observed > 0.8
    assert abs(test.null.mean()) < 0.03 and test.null.max() < 0.1
    assert test.p_value == pytest.approx(1 / 31)  # smallest attainable

    independent = _rng(18).normal(size=400)
    test = redundancy.permutation_test(model, independent, 30, _rng(19))
    assert test.p_value > 0.05
    assert test.p_value == pytest.approx((1 + (test.null >= test.observed).sum()) / 31)


def test_f_test_detects_linear_dependence_and_not_noise() -> None:
    u = _sphere(500, 5, 20)
    noise = _rng(21).normal(size=500)
    linear = redundancy.f_test(u, 2 * u[:, 0] - u[:, 1] + 0.5 * noise)
    assert linear.p_value < 1e-10 and linear.r2 > 0.8
    assert (linear.df1, linear.df2) == (5, 494)
    assert redundancy.f_test(u, noise).p_value > 0.01


def test_class_baseline_explains_class_driven_radius() -> None:
    rng = _rng(8)
    labels = rng.integers(0, 10, 2000)
    y = labels * 0.5 + 0.1 * rng.normal(size=labels.size)
    fit = redundancy.group_mean_cv(labels, y, 5, rng)
    assert fit.r2 > 0.95


def test_aurc_orders_oracle_random_and_adversarial_confidence() -> None:
    rng = _rng(9)
    errors = rng.random(5000) < 0.3
    oracle = numpy.where(errors, 0.0, 1.0) + 0.01 * rng.random(errors.size)
    random_conf = rng.random(errors.size)
    assert metrics.aurc(random_conf, errors) == pytest.approx(0.3, abs=0.02)
    assert metrics.aurc(oracle, errors) < 0.1
    assert metrics.aurc(-oracle, errors) > 0.6
    # Exact small case: errors ranked last -> risks 0, 0, 1/3.
    assert metrics.aurc([3, 2, 1], [False, False, True]) == pytest.approx(1 / 9)


def test_auroc_sanity_cases() -> None:
    assert metrics.auroc([3, 4, 1, 2], [True, True, False, False]) == 1.0
    assert metrics.auroc([1, 2, 3, 4], [True, True, False, False]) == 0.0
    assert metrics.auroc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    rng = _rng(10)
    shifted = numpy.concatenate([rng.normal(1, 1, 4000), rng.normal(0, 1, 4000)])
    positive = numpy.arange(8000) < 4000
    # Binormal AUROC with unit shift: Phi(1 / sqrt(2)) ~= 0.760.
    assert metrics.auroc(shifted, positive) == pytest.approx(0.760, abs=0.015)


def test_partial_spearman_removes_class_confound() -> None:
    rng = _rng(11)
    labels = rng.integers(0, 5, 3000)
    # x and y both depend on class only: raw correlation high, partial ~0.
    x = labels + 0.3 * rng.normal(size=labels.size)
    y = labels + 0.3 * rng.normal(size=labels.size)
    raw = metrics.spearman(x, y, num_bootstrap=100, rng=rng)
    partial = metrics.partial_spearman(x, y, labels, num_bootstrap=100, rng=rng)
    assert raw.value > 0.8
    assert raw.low <= raw.value <= raw.high
    assert abs(partial.value) < 0.06
    assert partial.low < 0 < partial.high


def test_knn_margin_and_probe_on_separable_clusters() -> None:
    rng = _rng(12)
    centers = numpy.eye(4, 16) * 5
    bank_labels = rng.integers(0, 4, 400)
    bank = centers[bank_labels] + rng.normal(size=(400, 16))
    query_labels = rng.integers(0, 4, 200)
    query = centers[query_labels] + rng.normal(size=(200, 16))
    votes = downstream.knn_votes(bank, bank_labels, query, 20, 4)
    numpy.testing.assert_allclose(votes.sum(axis=1), 1.0)
    assert downstream.accuracy(votes.argmax(axis=1), query_labels) > 0.95
    assert numpy.median(downstream.vote_margin(votes, query_labels)) > 0.8
    probe = downstream.LinearProbe(bank, bank_labels, 4)
    assert downstream.accuracy(probe.predict(query), query_labels) > 0.95
    assert downstream.entropy(probe.predict_proba(query)).mean() < 0.3


def test_view_consistency_and_center_view_selection() -> None:
    rng = _rng(13)
    base = rng.normal(size=(50, 1, 32))
    identical = numpy.repeat(base, 9, axis=1)
    numpy.testing.assert_allclose(features.view_consistency(identical), 1.0)
    noisy = rng.normal(size=(50, 9, 32))
    assert abs(features.view_consistency(noisy).mean()) < 0.05
    center = features.summarize(noisy, "center")
    numpy.testing.assert_allclose(
        center.rho_bar, numpy.log(numpy.linalg.norm(noisy[:, 0], axis=1))
    )
    assert features.summarize(noisy, "augmented").rho.shape == (50, 8)
