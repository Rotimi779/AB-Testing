"""Decision-rule tests. delta = 1.5 pp is a test fixture only, not project policy."""
import pytest

from statistical_tests import (
    CONFIG_REQUIRED,
    DECISION_STATES,
    GREENLIGHT,
    INCONCLUSIVE,
    INVALID,
    LOW_IMPACT,
    PROMISING,
    STOP_REVERT,
    classify_experiment_decision,
    describe_decision,
)


def decide(effect, lower, upper, p, delta, **kwargs):
    return classify_experiment_decision(
        effect_pp=effect,
        lower_ci_pp=lower,
        upper_ci_pp=upper,
        p_value=p,
        practical_threshold_pp=delta,
        **kwargs,
    )


@pytest.mark.parametrize(
    "effect, lower, upper, p, delta, expected",
    [
        pytest.param(2.5, 1.8, 3.2, 0.01, 1.5, GREENLIGHT, id="A-greenlight"),
        pytest.param(2.5, 0.8, 4.2, 0.01, 1.5, PROMISING, id="B-promising-significant"),
        pytest.param(2.5, -0.5, 5.5, 0.11, 2.0, PROMISING, id="C-promising-not-significant"),
        pytest.param(1.2, 0.3, 2.4, 0.01, 1.5, PROMISING, id="path-B-estimate-below-threshold"),
        pytest.param(0.7, 0.2, 1.2, 0.01, 1.5, LOW_IMPACT, id="D-low-impact"),
        pytest.param(1.0, -0.5, 3.0, 0.30, 2.0, INCONCLUSIVE, id="E-inconclusive-worthwhile-possible"),
        pytest.param(0.4, -0.5, 1.2, 0.40, 2.0, INCONCLUSIVE, id="F-inconclusive-low-upside"),
        pytest.param(-2.0, -3.0, -1.0, 0.01, 1.5, STOP_REVERT, id="G-stop-revert"),
        pytest.param(-0.5, -2.0, 1.0, 0.45, 1.5, INCONCLUSIVE, id="H-negative-not-significant"),
    ],
)
def test_decision_states(effect, lower, upper, p, delta, expected):
    assert decide(effect, lower, upper, p, delta)["decision"] == expected


def test_greenlight_fields():
    result = decide(2.5, 1.8, 3.2, 0.01, 1.5)
    assert result["statistically_significant"] is True
    assert result["positive_significant"] is True
    assert result["negative_significant"] is False
    assert result["estimate_meets_practical_threshold"] is True
    assert result["ci_clears_practical_threshold"] is True
    assert result["practical_value_still_plausible"] is True
    assert result["decision_reason"]


def test_low_upside_reason_is_flagged_but_label_is_canonical():
    result = decide(0.4, -0.5, 1.2, 0.40, 2.0)
    assert result["decision"] == INCONCLUSIVE
    assert "LOW EVIDENCE OF WORTHWHILE UPSIDE" in result["decision_reason"]


def test_validity_override():
    result = decide(2.5, 1.8, 3.2, 0.01, 1.5, experiment_valid=False)
    assert result["decision"] == INVALID


def test_validity_override_takes_priority_over_missing_threshold():
    assert decide(2.5, 1.8, 3.2, 0.01, None, experiment_valid=False)["decision"] == INVALID


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_missing_threshold_requires_config_and_is_not_invented(missing):
    result = decide(2.5, 1.8, 3.2, 0.01, missing)
    assert result["decision"] == CONFIG_REQUIRED
    # Statistical results are still reported...
    assert result["statistically_significant"] is True
    assert result["positive_significant"] is True
    # ...but no practical comparison is made against a substitute threshold.
    assert result["estimate_meets_practical_threshold"] is None
    assert result["ci_clears_practical_threshold"] is None
    assert result["practical_value_still_plausible"] is None


# --- Boundary conventions (spec section 5) ---------------------------------

def test_p_equal_to_alpha_is_significant():
    result = decide(2.5, 1.8, 3.2, 0.05, 1.5, alpha=0.05)
    assert result["statistically_significant"] is True
    assert result["decision"] == GREENLIGHT


def test_p_just_above_alpha_is_not_significant():
    assert decide(2.5, 1.8, 3.2, 0.0500001, 1.5)["statistically_significant"] is False


def test_lower_ci_equal_to_threshold_clears_threshold():
    result = decide(2.5, 1.5, 3.5, 0.01, 1.5)
    assert result["ci_clears_practical_threshold"] is True
    assert result["decision"] == GREENLIGHT


def test_upper_ci_equal_to_zero_is_not_entirely_negative():
    result = decide(-1.0, -2.0, 0.0, 0.04, 1.5)
    assert result["negative_significant"] is False
    assert result["decision"] == INCONCLUSIVE


def test_lower_ci_equal_to_zero_is_not_entirely_positive():
    result = decide(0.5, 0.0, 1.0, 0.04, 1.5)
    assert result["positive_significant"] is False
    assert result["decision"] == INCONCLUSIVE


# --- Input validation -------------------------------------------------------

@pytest.mark.parametrize("delta", [0.0, -1.5])
def test_non_positive_threshold_rejected(delta):
    with pytest.raises(ValueError):
        decide(2.5, 1.8, 3.2, 0.01, delta)


def test_inverted_ci_rejected():
    with pytest.raises(ValueError):
        decide(2.5, 3.2, 1.8, 0.01, 1.5)


# --- Summary / recommendation text ------------------------------------------

@pytest.mark.parametrize("decision", DECISION_STATES)
def test_every_state_has_text_and_no_run_until_significant_advice(decision):
    summary, recommendation = describe_decision(decision, 1.0, -0.5, 2.5, 0.2)
    assert summary and recommendation
    text = (summary + " " + recommendation).lower()
    assert "keep running" not in text
    assert "run longer" not in text


def test_unknown_decision_rejected():
    with pytest.raises(ValueError):
        describe_decision("KEEP RUNNING", 1.0, -0.5, 2.5, 0.2)


def test_inconclusive_text_does_not_claim_no_effect():
    _, recommendation = describe_decision(INCONCLUSIVE, 0.4, -0.5, 1.2, 0.4)
    assert "Do not infer no effect" in recommendation
