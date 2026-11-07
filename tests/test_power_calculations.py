"""Alpha and power must flow through the power, sample-size and MDE calculators."""
import pandas as pd
import pytest

from statistical_tests import (
    calculate_minimum_detectable_effect_from_results,
    calculate_sample_users_from_results,
    calculate_statistical_power_from_results,
)


@pytest.fixture
def summary():
    # 5.00% -> 5.75% with 10,000 users per group (the README example).
    return pd.DataFrame({
        "control_rate": [0.05],
        "treatment_rate": [0.0575],
        "control_size": [10000],
        "treatment_size": [10000],
        "lift_percent": [15.0],
    })


def power_of(summary, **kwargs):
    return float(calculate_statistical_power_from_results(summary, **kwargs)[0])


# --- Backward compatibility at alpha = 0.05, power = 0.80 ---------------------

def test_defaults_match_explicit_standard_values(summary):
    assert power_of(summary) == power_of(summary, alpha=0.05)
    assert calculate_sample_users_from_results(summary) == calculate_sample_users_from_results(
        summary, alpha=0.05, power=0.80)
    assert calculate_minimum_detectable_effect_from_results(summary) == calculate_minimum_detectable_effect_from_results(
        summary, alpha=0.05, power=0.80)


def test_defaults_reproduce_previously_documented_values(summary):
    # Values produced by the original hardcoded implementation (also shown in the README).
    assert power_of(summary) == pytest.approx(0.653, abs=0.001)
    required_users, total_required, _ = calculate_sample_users_from_results(summary)
    assert (required_users, total_required) == (14177, 28354)
    mde = calculate_minimum_detectable_effect_from_results(summary)
    assert mde["mde_pp"] == pytest.approx(0.90, abs=0.005)
    assert mde["mde_relative_lift_pct"] == pytest.approx(18.0, abs=0.05)


# --- Alpha changes the results -------------------------------------------------

def test_stricter_alpha_lowers_power(summary):
    assert power_of(summary, alpha=0.01) < power_of(summary, alpha=0.05) < power_of(summary, alpha=0.10)


def test_stricter_alpha_needs_more_users(summary):
    strict, _, _ = calculate_sample_users_from_results(summary, alpha=0.01)
    standard, _, _ = calculate_sample_users_from_results(summary, alpha=0.05)
    loose, _, _ = calculate_sample_users_from_results(summary, alpha=0.10)
    assert strict > standard > loose


def test_stricter_alpha_raises_mde(summary):
    strict = calculate_minimum_detectable_effect_from_results(summary, alpha=0.01)["mde_pp"]
    standard = calculate_minimum_detectable_effect_from_results(summary, alpha=0.05)["mde_pp"]
    loose = calculate_minimum_detectable_effect_from_results(summary, alpha=0.10)["mde_pp"]
    assert strict > standard > loose


# --- Desired power changes the results -----------------------------------------

def test_higher_power_needs_more_users(summary):
    low, _, _ = calculate_sample_users_from_results(summary, power=0.70)
    standard, _, _ = calculate_sample_users_from_results(summary, power=0.80)
    high, _, _ = calculate_sample_users_from_results(summary, power=0.90)
    assert low < standard < high


def test_higher_power_raises_mde(summary):
    low = calculate_minimum_detectable_effect_from_results(summary, power=0.70)["mde_pp"]
    standard = calculate_minimum_detectable_effect_from_results(summary, power=0.80)["mde_pp"]
    high = calculate_minimum_detectable_effect_from_results(summary, power=0.90)["mde_pp"]
    assert low < standard < high


def test_sample_size_and_power_are_consistent(summary):
    # Sizing for a target power, then computing power at that size, should return the target.
    for alpha, target in [(0.01, 0.90), (0.10, 0.70)]:
        required_users, _, _ = calculate_sample_users_from_results(summary, alpha=alpha, power=target)
        sized = summary.assign(control_size=[required_users], treatment_size=[required_users])
        assert power_of(sized, alpha=alpha) == pytest.approx(target, abs=0.001)


# --- Invalid inputs ------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, 1.0, -0.05, 1.5])
def test_invalid_alpha_rejected(summary, bad):
    with pytest.raises(ValueError):
        calculate_statistical_power_from_results(summary, alpha=bad)
    with pytest.raises(ValueError):
        calculate_sample_users_from_results(summary, alpha=bad)
    with pytest.raises(ValueError):
        calculate_minimum_detectable_effect_from_results(summary, alpha=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.8, 1.2])
def test_invalid_power_rejected(summary, bad):
    with pytest.raises(ValueError):
        calculate_sample_users_from_results(summary, power=bad)
    with pytest.raises(ValueError):
        calculate_minimum_detectable_effect_from_results(summary, power=bad)
