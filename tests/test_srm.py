import pandas as pd
import pytest

from statistical_tests import check_sample_ratio_mismatch, summarize_assignments


def test_balanced_allocation_passes():
    result = check_sample_ratio_mismatch(5000, 5000)
    assert result["srm_pass"] is True
    assert result["srm_p_value"] == pytest.approx(1.0)
    assert result["expected_control_n"] == 5000
    assert result["expected_treatment_n"] == 5000
    assert result["observed_control_n"] == 5000
    assert result["observed_treatment_n"] == 5000


def test_severe_imbalance_fails():
    result = check_sample_ratio_mismatch(5000, 5600)
    assert result["srm_pass"] is False
    assert result["srm_p_value"] < 0.001


def test_non_50_50_expected_allocation():
    # A 20/80 design observed as 2000/8000 matches the plan exactly.
    planned = check_sample_ratio_mismatch(2000, 8000, expected_treatment_share=0.8)
    assert planned["srm_pass"] is True
    assert planned["expected_control_n"] == pytest.approx(2000)
    assert planned["expected_treatment_n"] == pytest.approx(8000)
    # The same counts tested against a 50/50 plan are a severe mismatch.
    assert check_sample_ratio_mismatch(2000, 8000, expected_treatment_share=0.5)["srm_pass"] is False


def test_srm_alpha_is_configurable():
    # Mild imbalance with p of roughly 0.008: passes at 0.001, fails at 0.01.
    assert check_sample_ratio_mismatch(6891, 7204, srm_alpha=0.001)["srm_pass"] is True
    assert check_sample_ratio_mismatch(6891, 7204, srm_alpha=0.01)["srm_pass"] is False


@pytest.mark.parametrize(
    "control_n, treatment_n, share",
    [
        (0, 0, 0.5),      # zero total users
        (100, 100, 0.0),  # share outside (0, 1)
        (100, 100, 1.0),
        (100, 100, 1.5),
        (-1, 100, 0.5),   # negative count
    ],
)
def test_invalid_inputs_raise(control_n, treatment_n, share):
    with pytest.raises(ValueError):
        check_sample_ratio_mismatch(control_n, treatment_n, expected_treatment_share=share)


def test_assignment_summary_counts_unique_users_and_cross_variant_assignments():
    assignments = pd.DataFrame({
        "experiment_id": [1, 1, 1, 1, 1, 2, 2],
        "user_id": [10, 11, 12, 12, 10, 20, 21],
        "variant": ["control", "treatment", "control", "treatment", "control", "control", "treatment"],
    })
    summary = summarize_assignments(assignments)
    # User 10 has a duplicate control row (counted once); user 12 is in both variants.
    assert summary.loc[1, "control_n"] == 2
    assert summary.loc[1, "treatment_n"] == 2
    assert summary.loc[1, "cross_variant_users"] == 1
    assert summary.loc[2, "cross_variant_users"] == 0


def test_assignment_summary_rejects_unknown_variants():
    assignments = pd.DataFrame({"experiment_id": [1], "user_id": [1], "variant": ["holdout"]})
    with pytest.raises(ValueError):
        summarize_assignments(assignments)
