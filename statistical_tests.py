"""
Statistical tests, decision rules and the experiment-summary pipeline.

Importing this module is side-effect free: it only defines functions and constants.
The batch analysis runs through ``build_experiment_summary()``. To recompute results
AND refresh the SQLite table the dashboard reads, run::

    python refresh_results.py

Units convention used throughout this module:
- ``*_rate``, ``difference``, ``lower_ci``, ``upper_ci`` are proportions (0.015 == 1.5%).
- Anything ending in ``_pp`` is in percentage points (1.5 == 1.5 pp).
"""
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize


DATA_DIR = Path("data")
SUMMARY_CSV_NAME = "summary_and_analysis.csv"
POLICY_CSV_NAME = "experiment_policy.csv"

# ---------------------------------------------------------------------------
# Canonical decision labels (downstream code and the dashboard depend on these)
# ---------------------------------------------------------------------------
GREENLIGHT = "GREENLIGHT"
PROMISING = "PROMISING / NEED MORE EVIDENCE"
LOW_IMPACT = "LOW IMPACT"
INCONCLUSIVE = "INCONCLUSIVE / HOLD"
STOP_REVERT = "STOP / REVERT"
INVALID = "INVALID / NO DECISION"
CONFIG_REQUIRED = "CONFIG REQUIRED / NO PRACTICAL DECISION"

DECISION_STATES = (
    GREENLIGHT,
    PROMISING,
    LOW_IMPACT,
    INCONCLUSIVE,
    STOP_REVERT,
    INVALID,
    CONFIG_REQUIRED,
)

VALIDITY_PASS = "PASS"
VALIDITY_FAIL = "FAIL"

# The confirmatory experiment decision pipeline currently uses alpha = 0.05
# because the reported outcome confidence interval is fixed at 95% (critical value 1.96).
# The standalone planning calculators for power, sample size, and MDE support
# configurable alpha and target-power values.
SUPPORTED_ALPHA = 0.05

DEFAULT_SRM_ALPHA = 0.001


# ---------------------------------------------------------------------------
# Two-proportion z-test (unchanged)
# ---------------------------------------------------------------------------
def two_proportion_ztest(n1, x1, n2, x2):
    """
    Perform two-proportion z-test.

    Parameters:
    n1: sample size for group 1 (control)
    x1: number of successes in group 1 (control conversions)
    n2: sample size for group 2 (treatment)
    x2: number of successes in group 2 (treatment conversions)

    Returns:
    dict with z_score, p_value, lower confidence interval, upper confidence interval,
    lift_percent and difference. ``difference``, ``lower_ci`` and ``upper_ci`` are
    proportions (treatment minus control), not percentage points.
    """

    if n1 == 0 or n2 == 0:
        raise ValueError("Sample sizes n1 and n2 must be greater than 0.")

    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    if p_pool == 0 or p_pool == 1:
        return {'z_score': 0.0,'p_value': 1.0,'significant': False,'lower_ci': 0.0,'upper_ci': 0.0,'lift_percent': 0.0, 'difference': 0.0}



    standard_error = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if standard_error == 0:
        z_score = 0.0
        p_value_two_tail = 1.0
    else:
        z_score = (p2 - p1) / standard_error
        p_value_two_tail = 2 * stats.norm.sf(abs(z_score))


    difference = p2 - p1
    standard_error_diff = sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
    confidence_interval_lower = difference - 1.96 * standard_error_diff
    confidence_interval_upper = difference + 1.96 * standard_error_diff
    if p1 == 0:
        lift_percent = float("inf") if p2 > 0 else 0.0
    else:
        lift_percent = round((p2 - p1) / p1 * 100, 2)
    return {'z_score':z_score, 'p_value':p_value_two_tail,  'lower_ci': confidence_interval_lower, 'upper_ci': confidence_interval_upper, 'lift_percent':lift_percent,
            'difference':difference}


# ---------------------------------------------------------------------------
# Power, sample size and MDE (DataFrame-based API)
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def _validate_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")


def _validate_power(power):
    if not 0 < power < 1:
        raise ValueError("power must be strictly between 0 and 1.")


def calculate_statistical_power_from_results(summary_df, alpha=DEFAULT_ALPHA):
    """
    Power (two-sided, significance level ``alpha``) to detect the difference between
    ``control_rate`` and ``treatment_rate`` with the given group sizes.

    For planning, pass the effect you want to detect as ``treatment_rate``. When it is
    called with an experiment's observed rates, the result is power at the observed
    effect size. That is a function of the p-value and is not independent evidence that
    the experiment was reliable.

    ``summary_df`` is a one-row DataFrame with columns control_rate, treatment_rate,
    control_size and treatment_size. Returns a length-1 array.
    """
    _validate_alpha(alpha)
    p1 = summary_df['control_rate']
    p2 = summary_df['treatment_rate']
    effect_size = proportion_effectsize(p2, p1).iloc[0]
    nobs1 = summary_df['control_size']
    control_n = summary_df['control_size'].iloc[0]
    treatment_n = summary_df['treatment_size'].iloc[0]
    ratio = treatment_n / control_n
    power = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, nobs1=nobs1, ratio=ratio, alternative='two-sided')
    return power


def calculate_sample_users_from_results(summary_df, alpha=DEFAULT_ALPHA, power=DEFAULT_POWER):
    """
    Calculates the number of users per group required to detect a lift, the total number of users based on the ratio and the multiplier to reach
    the required users per group (two-sided, significance level ``alpha``, target ``power``).

    ``summary_df`` is a one-row DataFrame with columns control_rate, treatment_rate,
    control_size and treatment_size. Returns ``(required_users, total_required, multiplier)``.
    """
    _validate_alpha(alpha)
    _validate_power(power)

    p1 = summary_df['control_rate'].iloc[0]
    p2 = summary_df['treatment_rate'].iloc[0]
    control_n = summary_df['control_size'].iloc[0]
    treatment_n = summary_df['treatment_size'].iloc[0]
    effect_size = proportion_effectsize(p2, p1)

    # Handle zero or very small effect sizes
    if effect_size == 0 or abs(effect_size) < 0.001:
        # If no meaningful effect, return large sample requirement or current sample
        return control_n, control_n + treatment_n, 1.0

    ratio = treatment_n / control_n

    #required_users: required users per group to detect the effect at the target power
    try:
        required_users = int(round(zt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio, alternative='two-sided'), 0))
    except ValueError:
        # If power calculation fails, return current sample
        return control_n, control_n + treatment_n, 1.0

    total_current = control_n + treatment_n
    total_required = int(round(required_users * (1 + ratio),0))

    #multiplier:tells you how many times bigger your current per-group sample needs to be (relative to your current control group size)
    # to have enough power to detect the observed effect.
    multiplier = round(required_users / control_n,1)
    return required_users, total_required, multiplier


def calculate_minimum_detectable_effect_from_results(summary_df, alpha=DEFAULT_ALPHA, power=DEFAULT_POWER):
    """
    Calculates the minimum lift you could have detected from control conversion to treatment conversion
    (two-sided, significance level ``alpha``, target ``power``).

    ``summary_df`` is a one-row DataFrame with columns control_rate, control_size,
    treatment_size and lift_percent. Returns a dict with ``mde_relative_lift_pct``,
    ``mde_pp`` (percentage points) and ``actual_lift_pct``.
    """
    _validate_alpha(alpha)
    _validate_power(power)
    control_rate = float(summary_df['control_rate'].iloc[0])
    control_n = int(summary_df['control_size'].iloc[0])
    treatment_n = int(summary_df['treatment_size'].iloc[0])
    ratio = treatment_n / control_n

    # 1) Solve required Cohen's h given n, power, alpha
    effect_size = zt_ind_solve_power(
        effect_size=None,
        nobs1=control_n,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative='two-sided'
    )

    # Guard: if effect_size is 0 (can happen with extreme inputs)
    if effect_size <= 0:
        return {
            "mde_relative_lift_pct": 0.0,
            "mde_pp": 0.0,
            "actual_lift_pct": float(summary_df['lift_percent'].iloc[0]),
        }

    # 2) Solve for p2 such that h(p2, p1) = effect_size
    eps = 1e-12
    p1 = min(max(control_rate, eps), 1 - eps)

    def f(p2):
        p2 = min(max(p2, eps), 1 - eps)
        h = 2*np.arcsin(np.sqrt(p2)) - 2*np.arcsin(np.sqrt(p1))
        return h - effect_size

    # Bracket for brentq: near 0 to near 1
    a, b = eps, 1 - eps

    # Ensure there is a sign change; if not, fall back gracefully
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        # Couldn't bracket a root; return NaN-like or fallback info
        return {
            "mde_relative_lift_pct": float("nan"),
            "mde_pp": 0.0,
            "actual_lift_pct": float(summary_df['lift_percent'].iloc[0]),
        }

    mde_treatment_rate = brentq(f, a, b)

    mde_pp = (mde_treatment_rate - control_rate) * 100
    # 3) Convert to relative lift
    if control_rate == 0:
        mde_relative_lift = float("inf") if mde_treatment_rate > 0 else 0.0
    else:
        mde_relative_lift = (mde_treatment_rate - control_rate) / control_rate * 100

    return {
        "mde_relative_lift_pct": float(mde_relative_lift),
        "mde_pp": float(mde_pp),
        "actual_lift_pct": float(summary_df['lift_percent'].iloc[0]),
    }


# ---------------------------------------------------------------------------
# Decision rule (pure function, no pandas / SQLite / Streamlit)
# ---------------------------------------------------------------------------
def _threshold_is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def classify_experiment_decision(
    effect_pp: float,
    lower_ci_pp: float,
    upper_ci_pp: float,
    p_value: float,
    practical_threshold_pp,
    alpha: float = 0.05,
    experiment_valid: bool = True,
) -> dict:
    """
    Classify an experiment into one of the canonical decision states.

    All effect inputs are in percentage points. ``practical_threshold_pp`` (delta) is the
    smallest positive improvement the business considers worth shipping, chosen before
    looking at the final result. Pass ``None``/NaN when it has not been configured; the
    statistical fields are still returned but the decision is CONFIG REQUIRED.

    Only CI endpoints are compared with 0 and delta. Boundary conventions:
    - p_value <= alpha          -> statistically significant
    - lower_ci_pp >= delta      -> CI clears the practical threshold
    - lower_ci_pp > 0           -> CI entirely positive (== 0 is not)
    - upper_ci_pp < 0           -> CI entirely harmful (== 0 is not)
    - upper_ci_pp < delta       -> CI entirely below the practical threshold

    Order of evaluation: validity -> configuration -> GREENLIGHT -> STOP / REVERT
    -> PROMISING -> LOW IMPACT -> INCONCLUSIVE / HOLD.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0 <= p_value <= 1:
        raise ValueError("p_value must be between 0 and 1.")
    if lower_ci_pp > upper_ci_pp:
        raise ValueError("lower_ci_pp must not exceed upper_ci_pp.")

    threshold_missing = _threshold_is_missing(practical_threshold_pp)
    if not threshold_missing and practical_threshold_pp <= 0:
        raise ValueError("practical_threshold_pp must be a positive number of percentage points.")

    statistically_significant = p_value <= alpha
    positive_significant = statistically_significant and lower_ci_pp > 0
    negative_significant = statistically_significant and upper_ci_pp < 0

    if threshold_missing:
        estimate_meets = None
        ci_clears = None
        practical_plausible = None
    else:
        delta = practical_threshold_pp
        estimate_meets = effect_pp >= delta
        ci_clears = lower_ci_pp >= delta
        practical_plausible = upper_ci_pp >= delta

    result = {
        "decision": None,
        "statistically_significant": statistically_significant,
        "positive_significant": positive_significant,
        "negative_significant": negative_significant,
        "estimate_meets_practical_threshold": estimate_meets,
        "ci_clears_practical_threshold": ci_clears,
        "practical_value_still_plausible": practical_plausible,
        "decision_reason": None,
    }

    if not experiment_valid:
        result["decision"] = INVALID
        result["decision_reason"] = (
            "The experiment failed a validity check (for example sample-ratio mismatch or "
            "cross-variant assignment). The effect, CI and p-value must not drive a launch decision."
        )
        return result

    if threshold_missing:
        result["decision"] = CONFIG_REQUIRED
        result["decision_reason"] = (
            "No practical threshold is configured for this experiment, so a practical launch "
            "decision cannot be made. The threshold must be chosen independently of the observed result."
        )
        return result

    if statistically_significant and ci_clears:
        result["decision"] = GREENLIGHT
        result["decision_reason"] = (
            "Statistically significant and even the lower CI bound clears the practical threshold."
        )
    elif negative_significant:
        result["decision"] = STOP_REVERT
        result["decision_reason"] = (
            "Statistically significant harm: the entire CI is below zero."
        )
    elif estimate_meets:
        result["decision"] = PROMISING
        if statistically_significant:
            result["decision_reason"] = (
                "The point estimate meets the practical threshold but the lower CI bound does not, "
                "so practical value is not established."
            )
        else:
            result["decision_reason"] = (
                "The point estimate meets the practical threshold but the result is not statistically "
                "significant; zero and sub-threshold effects remain compatible with the data."
            )
    elif positive_significant and practical_plausible:
        result["decision"] = PROMISING
        result["decision_reason"] = (
            "A positive effect appears real and the CI still includes worthwhile effects, but the "
            "point estimate is below the practical threshold."
        )
    elif positive_significant:
        result["decision"] = LOW_IMPACT
        result["decision_reason"] = (
            "A positive effect appears real, but even the upper CI bound is below the practical threshold."
        )
    else:
        result["decision"] = INCONCLUSIVE
        if practical_plausible:
            result["decision_reason"] = (
                "Not enough evidence of a positive or harmful effect, and the point estimate is below the "
                "practical threshold. A worthwhile effect remains compatible with the data."
            )
        else:
            result["decision_reason"] = (
                "INCONCLUSIVE / HOLD — LOW EVIDENCE OF WORTHWHILE UPSIDE: no clear evidence of an effect "
                "and even the upper CI bound is below the practical threshold."
            )
    return result


def describe_decision(decision, effect_pp, lower_ci_pp, upper_ci_pp, p_value):
    """Return ``(summary, recommendation)`` text for a decision state."""
    stats_text = (
        f"Observed effect {effect_pp:+.2f} pp (95% CI [{lower_ci_pp:.2f}, {upper_ci_pp:.2f}] pp, p={p_value:.3g})."
    )
    if decision == GREENLIGHT:
        summary = stats_text + " Statistically significant and the whole CI clears the practical threshold."
        recommendation = "Roll out the change (or ramp up rollout). Monitor key metrics after launch."
    elif decision == PROMISING:
        summary = stats_text + " A worthwhile improvement is plausible, but practical value is not established."
        recommendation = (
            "Do not launch on this result alone. If further evidence is needed, plan a follow-up experiment "
            "sized for the practical threshold, or continue only if doing so is consistent with the original "
            "stopping rule / sequential design."
        )
    elif decision == LOW_IMPACT:
        summary = stats_text + " The effect appears positive, but even the upper CI bound is below the practical threshold."
        recommendation = (
            "The plausible improvement is smaller than the prespecified minimum worthwhile effect. "
            "Deprioritize unless the decision policy is revised independently of this result."
        )
    elif decision == INCONCLUSIVE:
        summary = stats_text + " Inconclusive under the prespecified analysis."
        recommendation = (
            "The current result is inconclusive under the prespecified analysis. Do not infer no effect. "
            "If further evidence is needed, plan a follow-up or continue only if doing so is consistent with "
            "the original stopping rule / sequential design."
        )
    elif decision == STOP_REVERT:
        summary = stats_text + " Statistically significant degradation."
        recommendation = "Stop the experiment and revert. Investigate why performance dropped."
    elif decision == INVALID:
        summary = "The experiment failed a validity check, so the treatment effect should not be interpreted."
        recommendation = (
            "Investigate randomization and assignment instrumentation before interpreting the treatment effect. "
            "Do not use this result for a launch decision."
        )
    elif decision == CONFIG_REQUIRED:
        summary = stats_text + " No practical threshold is configured, so no practical launch decision is made."
        recommendation = (
            f"Set practical_threshold_pp for this experiment in data/{POLICY_CSV_NAME}, chosen independently of "
            "the observed result, then run `python refresh_results.py`."
        )
    else:
        raise ValueError(f"Unknown decision state: {decision!r}")
    return summary, recommendation


# ---------------------------------------------------------------------------
# Sample ratio mismatch and assignment integrity
# ---------------------------------------------------------------------------
def check_sample_ratio_mismatch(
    control_n: int,
    treatment_n: int,
    expected_treatment_share: float = 0.5,
    srm_alpha: float = DEFAULT_SRM_ALPHA,
) -> dict:
    """
    Chi-square goodness-of-fit test of observed vs expected allocation for two arms.

    ``srm_alpha`` is separate from the outcome alpha. The default of 0.001 is strict on
    purpose: SRM is a data-quality alarm, and a failure blocks the launch decision.
    """
    if control_n < 0 or treatment_n < 0:
        raise ValueError("Group sizes must be non-negative.")
    total = control_n + treatment_n
    if total <= 0:
        raise ValueError("Total number of assigned users must be greater than 0.")
    if not 0 < expected_treatment_share < 1:
        raise ValueError("expected_treatment_share must be strictly between 0 and 1.")
    if not 0 < srm_alpha < 1:
        raise ValueError("srm_alpha must be strictly between 0 and 1.")

    expected_treatment_n = total * expected_treatment_share
    expected_control_n = total - expected_treatment_n
    chi2, p_value = stats.chisquare(
        [control_n, treatment_n], f_exp=[expected_control_n, expected_treatment_n]
    )
    p_value = float(p_value)
    return {
        "srm_chi2": float(chi2),
        "srm_p_value": p_value,
        "srm_pass": p_value >= srm_alpha,
        "expected_control_n": float(expected_control_n),
        "expected_treatment_n": float(expected_treatment_n),
        "observed_control_n": int(control_n),
        "observed_treatment_n": int(treatment_n),
    }


def summarize_assignments(assignments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per experiment: unique users assigned to control and treatment, and the number of
    users assigned to both variants of the same experiment (an integrity defect).

    Expects columns experiment_id, user_id and variant. Returns a DataFrame indexed by
    experiment_id with columns control_n, treatment_n and cross_variant_users.
    """
    required = {"experiment_id", "user_id", "variant"}
    missing = required - set(assignments_df.columns)
    if missing:
        raise ValueError(f"Assignment data is missing columns: {sorted(missing)}")
    unknown = set(assignments_df["variant"].unique()) - {"control", "treatment"}
    if unknown:
        raise ValueError(f"Unexpected variant labels in assignment data: {sorted(unknown)}")

    counts = (
        assignments_df.groupby(["experiment_id", "variant"])["user_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=["control", "treatment"], fill_value=0)
        .rename(columns={"control": "control_n", "treatment": "treatment_n"})
    )
    variants_per_user = assignments_df.groupby(["experiment_id", "user_id"])["variant"].nunique()
    cross = (variants_per_user > 1).groupby(level="experiment_id").sum().rename("cross_variant_users")
    out = counts.join(cross, how="left").fillna({"cross_variant_users": 0})
    return out.astype(int)


# ---------------------------------------------------------------------------
# Experiment policy configuration
# ---------------------------------------------------------------------------
def load_experiment_policy(path=DATA_DIR / POLICY_CSV_NAME) -> pd.DataFrame:
    """
    Load the per-experiment decision policy, indexed by experiment_id.

    Columns: practical_threshold_pp (blank = not configured), alpha,
    expected_treatment_share, srm_alpha. Thresholds are never filled in automatically.
    """
    policy = pd.read_csv(path)
    required = ["experiment_id", "practical_threshold_pp", "alpha", "expected_treatment_share", "srm_alpha"]
    missing = [c for c in required if c not in policy.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if policy["experiment_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate experiment_id rows.")

    policy = policy.set_index("experiment_id")
    thresholds = policy["practical_threshold_pp"].dropna()
    if (thresholds <= 0).any():
        raise ValueError("practical_threshold_pp must be positive (percentage points) when configured.")
    if not np.allclose(policy["alpha"], SUPPORTED_ALPHA):
        raise ValueError(
            "Only alpha = 0.05 is supported for the confirmatory experiment decision pipeline "
            "because the reported outcome confidence interval is fixed at 95%. "
            "The standalone power, sample-size, and MDE planning calculators support configurable alpha values."
        )
    share = policy["expected_treatment_share"]
    if share.isna().any() or not ((share > 0) & (share < 1)).all():
        raise ValueError("expected_treatment_share must be strictly between 0 and 1.")
    srm_alpha = policy["srm_alpha"]
    if srm_alpha.isna().any() or not ((srm_alpha > 0) & (srm_alpha < 1)).all():
        raise ValueError("srm_alpha must be strictly between 0 and 1.")
    return policy


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------
def _as_float(value) -> float:
    return float(np.asarray(value).ravel()[0])


def build_experiment_summary(data_dir=DATA_DIR) -> pd.DataFrame:
    """Compute the per-experiment summary that the dashboard reads."""
    data_dir = Path(data_dir)
    user_sessions_df = pd.read_csv(data_dir / "user_sessions.csv")
    experiments_df = pd.read_csv(data_dir / "experiments.csv")
    assignments_df = pd.read_csv(data_dir / "experiment_assignments.csv")
    policy = load_experiment_policy(data_dir / POLICY_CSV_NAME)
    assignment_counts = summarize_assignments(assignments_df)

    rows = []
    for experiment_id, experiment_name in zip(experiments_df["experiment_id"], experiments_df["experiment_name"]):
        if experiment_id not in policy.index:
            raise ValueError(
                f"Experiment {experiment_id} ({experiment_name}) has no row in data/{POLICY_CSV_NAME}."
            )
        exp_policy = policy.loc[experiment_id]
        alpha = float(exp_policy["alpha"])
        threshold_pp = exp_policy["practical_threshold_pp"]
        threshold_pp = None if pd.isna(threshold_pp) else float(threshold_pp)
        expected_share = float(exp_policy["expected_treatment_share"])
        srm_alpha = float(exp_policy["srm_alpha"])

        # Outcome analysis (user-level conversions from sessions; unchanged method)
        df = user_sessions_df[user_sessions_df["experiment_id"] == experiment_id]
        control_conv = df[df["variant"] == "control"].groupby("user_id")["converted"].max()
        treatment_conv = df[df["variant"] == "treatment"].groupby("user_id")["converted"].max()
        control_n = len(control_conv)
        treatment_n = len(treatment_conv)
        control_x = control_conv.sum()
        treatment_x = treatment_conv.sum()
        control_rate = round(control_x / control_n, 4)
        treatment_rate = round(treatment_x / treatment_n, 4)

        results = two_proportion_ztest(control_n, control_x, treatment_n, treatment_x)
        diff_pp = results["difference"] * 100
        lower_pp = results["lower_ci"] * 100
        upper_pp = results["upper_ci"] * 100
        p_value = float(results["p_value"])

        # Validity: SRM and cross-variant assignment from the authoritative assignment data
        if experiment_id in assignment_counts.index:
            assigned = assignment_counts.loc[experiment_id]
            srm = check_sample_ratio_mismatch(
                int(assigned["control_n"]), int(assigned["treatment_n"]), expected_share, srm_alpha
            )
            cross_variant_users = int(assigned["cross_variant_users"])
            reasons = []
            if not srm["srm_pass"]:
                reasons.append(
                    f"Sample ratio mismatch (p={srm['srm_p_value']:.3g} < SRM alpha {srm_alpha:g})."
                )
            if cross_variant_users > 0:
                reasons.append(f"{cross_variant_users} user(s) assigned to both variants.")
        else:
            srm = {"srm_p_value": np.nan, "srm_pass": False,
                   "expected_control_n": np.nan, "expected_treatment_n": np.nan,
                   "observed_control_n": 0, "observed_treatment_n": 0}
            cross_variant_users = 0
            reasons = ["No assignment records found for this experiment."]
        experiment_valid = not reasons
        validity_status = VALIDITY_PASS if experiment_valid else VALIDITY_FAIL

        decision_info = classify_experiment_decision(
            effect_pp=diff_pp,
            lower_ci_pp=lower_pp,
            upper_ci_pp=upper_pp,
            p_value=p_value,
            practical_threshold_pp=threshold_pp,
            alpha=alpha,
            experiment_valid=experiment_valid,
        )
        summary, recommendation = describe_decision(
            decision_info["decision"], diff_pp, lower_pp, upper_pp, p_value
        )

        # Sensitivity (existing calculators)
        calc_input = pd.DataFrame({
            "control_rate": [control_rate],
            "treatment_rate": [treatment_rate],
            "control_size": [control_n],
            "treatment_size": [treatment_n],
            "lift_percent": [results["lift_percent"]],
        })
        power = _as_float(calculate_statistical_power_from_results(calc_input, alpha=alpha))
        required_users, total_required, multiplier = calculate_sample_users_from_results(
            calc_input, alpha=alpha, power=DEFAULT_POWER)
        mde = calculate_minimum_detectable_effect_from_results(calc_input, alpha=alpha, power=DEFAULT_POWER)
        mde_exceeds = None if threshold_pp is None else bool(mde["mde_pp"] > threshold_pp)

        rows.append({
            "experiment_id": int(experiment_id),
            "experiment_name": experiment_name,
            "control_rate": control_rate,
            "control_size": control_n,
            "treatment_rate": treatment_rate,
            "treatment_size": treatment_n,
            "lift_percent": results["lift_percent"],
            "difference_pp": diff_pp,
            "z_score": results["z_score"],
            "p_value": p_value,
            "alpha": alpha,
            "statistically_significant": decision_info["statistically_significant"],
            "lower_ci": round(results["lower_ci"], 6),
            "upper_ci": round(results["upper_ci"], 6),
            "lower_ci_pp": lower_pp,
            "upper_ci_pp": upper_pp,
            "practical_threshold_pp": threshold_pp,
            "estimate_meets_practical_threshold": decision_info["estimate_meets_practical_threshold"],
            "ci_clears_practical_threshold": decision_info["ci_clears_practical_threshold"],
            "decision": decision_info["decision"],
            "decision_reason": decision_info["decision_reason"],
            "summary": summary,
            "recommendation": recommendation,
            "validity_status": validity_status,
            "validity_reason": " ".join(reasons),
            "srm_p_value": srm["srm_p_value"],
            "srm_pass": srm["srm_pass"],
            "srm_alpha": srm_alpha,
            "expected_treatment_share": expected_share,
            "assigned_control_n": srm["observed_control_n"],
            "assigned_treatment_n": srm["observed_treatment_n"],
            "expected_control_n": srm["expected_control_n"],
            "expected_treatment_n": srm["expected_treatment_n"],
            "cross_variant_users": cross_variant_users,
            "power": power,
            "required_users": float(required_users),
            "total_required": float(total_required),
            "multiplier": float(multiplier),
            "mde_pp": mde["mde_pp"],
            "mde_relative_lift_pct": mde["mde_relative_lift_pct"],
            "mde_exceeds_practical_threshold": mde_exceeds,
            "actual_lift_pct": mde["actual_lift_pct"],
        })

    return pd.DataFrame(rows)


def write_summary_csv(summary_df: pd.DataFrame, data_dir=DATA_DIR) -> Path:
    path = Path(data_dir) / SUMMARY_CSV_NAME
    summary_df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    summary_df = build_experiment_summary()
    path = write_summary_csv(summary_df)
    print(summary_df[["experiment_name", "difference_pp", "p_value", "decision", "validity_status"]])
    print(f"\nWrote {path}. Run `python refresh_results.py` to also refresh the SQLite table used by the dashboard.")
