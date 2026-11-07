from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statistical_tests import calculate_statistical_power_from_results,calculate_sample_users_from_results,calculate_minimum_detectable_effect_from_results
from statistical_tests import (
    GREENLIGHT, PROMISING, LOW_IMPACT, INCONCLUSIVE, STOP_REVERT, INVALID, CONFIG_REQUIRED,
    VALIDITY_PASS,
)

QUERIES = Path("queries")
DB_PATH = "ab_testing.db"

def load_query(filename: str) -> str:
    return (QUERIES / filename).read_text(encoding="utf-8")

def run_query_df(query: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)

@st.cache_data
def get_df(query_name: str, params: tuple):
    query = load_query(query_name)
    return run_query_df(query, params)

def has_value(value) -> bool:
    """True when a value read from SQLite is present (not NULL/NaN)."""
    return value is not None and not pd.isna(value)


#For confidence interval plots. Inputs are proportions; the plot is in percentage points.
#Only the CI endpoints are compared with 0 and the practical threshold; nothing inside the
#interval is shaded, because a confidence interval is not a probability distribution.
def ci_plot(difference, lower_ci, upper_ci, p_value, practical_threshold_pp=None, alpha=0.05):
    diff_pp  = difference * 100
    low_pp   = lower_ci * 100
    high_pp  = upper_ci * 100
    threshold_set = has_value(practical_threshold_pp)

    significant = p_value is not None and p_value <= alpha
    color = "seagreen" if significant else "darkorange"
    status = "Statistically significant" if significant else "Not statistically significant"

    xmin = min(low_pp, 0, diff_pp, practical_threshold_pp if threshold_set else 0)
    xmax = max(high_pp, 0, diff_pp, practical_threshold_pp if threshold_set else 0)
    span = xmax - xmin
    pad = max(0.25, span * 0.15)
    x_range = [xmin - pad, xmax + pad]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[diff_pp],
        y=["Lift"],
        mode="markers",
        marker=dict(size=12, color=color),
        error_x=dict(
            type="data",
            symmetric=False,
            array=[max(0, high_pp - diff_pp)],
            arrayminus=[max(0, diff_pp - low_pp)],
            thickness=2,
            width=6,
            color=color
        ),
        hovertemplate=(
            f"Status: {status}<br>"
            "Lift: %{x:.2f} pp<br>"
            f"95% CI: [{low_pp:.2f}, {high_pp:.2f}] pp<br>"
            + (f"p-value: {p_value:.4g}<br>" if p_value is not None else "")
            + "<extra></extra>"
        )
    ))

    # Reference line at 0 (no effect)
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="0 pp (no effect)", annotation_position="bottom left",
                  annotation_font_color="#90a4ae")

    # Reference line at the practical threshold (only when configured)
    if threshold_set:
        fig.add_vline(x=practical_threshold_pp, line_dash="dot", line_width=2, line_color="#00d4ff",
                      annotation_text=f"Practical threshold (demo) {practical_threshold_pp:.2f} pp",
                      annotation_position="top right", annotation_font_color="#00d4ff")

    annotation_lines = [
        f"{status}",
        f"Lift: {diff_pp:+.2f} pp",
        f"95% CI: [{low_pp:+.2f}, {high_pp:+.2f}] pp",
    ]

    if p_value is not None:
        annotation_lines.append(f"p = {p_value:.4g}")

    fig.add_annotation(
        x=diff_pp,
        y="Lift",
        text="<br>".join(annotation_lines),
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-60,
        bgcolor="rgba(13,17,23,0.85)",
        bordercolor=color,
        borderwidth=1,
        font=dict(color="#ffffff")
    )

    fig.update_layout(
        xaxis_title="Percentage points (pp)",
        yaxis_title="",
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#cfd8dc'),
        xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='#1e2a3a',
            showline=True, linewidth=1, linecolor='#2d3548',
            tickfont=dict(color='#90a4ae')
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color='#90a4ae')
        )
    )
    fig.update_xaxes(range=x_range, zeroline=False)
    return fig


st.set_page_config(page_title="A/B Testing Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark sleek styling
st.markdown("""
    <style>
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #00d4ff;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: inherit;
        opacity: 0.65;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .decision-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .decision-greenlight {
        background: linear-gradient(135deg, #0a2a1a 0%, #0d3320 100%);
        border-left: 5px solid #00e676;
    }
    .decision-warning {
        background: linear-gradient(135deg, #1a1a00 0%, #2a2200 100%);
        border-left: 5px solid #ffd600;
    }
    .decision-danger {
        background: linear-gradient(135deg, #2a0a0a 0%, #330d0d 100%);
        border-left: 5px solid #ff1744;
    }
    .decision-promising {
        background: linear-gradient(135deg, #0a1a2a 0%, #0d2033 100%);
        border-left: 5px solid #40c4ff;
    }
    .decision-neutral {
        background: linear-gradient(135deg, #1a1a1a 0%, #222222 100%);
        border-left: 5px solid #ffab40;
    }
    .decision-invalid {
        background: linear-gradient(135deg, #1f0a2a 0%, #2a0d33 100%);
        border-left: 5px solid #d500f9;
    }
    .decision-config {
        background: linear-gradient(135deg, #1a1f2e 0%, #232a3b 100%);
        border-left: 5px solid #90a4ae;
    }
    .metric-card {
        border-radius: 8px;
        padding: 1.25rem;
        background: #1a1f2e;
        border: 1px solid #2d3548;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
col_title, col_selector = st.columns([3, 1], vertical_alignment="center")

with col_title:
    st.markdown('<div class="header-title">📊 A/B Testing Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Real-time experiment monitoring & statistical insights</div>', unsafe_allow_html=True)

experiment_mapping_dict = {
    'Checkout Button Color': 1,
    'Pricing Display Test': 2,
    'Email Subject Line': 3,
    'Product Page Layout': 4,
    'Free Shipping Threshold': 5,
    'Search Result Sorting': 6,
    'Checkout Form Fields': 7,
    'Recommendation Algorithm': 8,
    'Cart Discount Display': 9,
    'Mobile App Onboarding': 10
}

with col_selector:
    experiment = st.selectbox(
        'Select Experiment',
        list(experiment_mapping_dict.keys()),
        label_visibility="collapsed"
    )

selected_experiment = experiment_mapping_dict[experiment]

experiments_results_summary = 'experiments_results_summary.txt'
experiments_results_summary_df = get_df(experiments_results_summary, (selected_experiment,)).iloc[0]

# Extract key metrics
lift = experiments_results_summary_df['lift_percent']
p_value = experiments_results_summary_df["p_value"]
z_score = experiments_results_summary_df['z_score']
decision = experiments_results_summary_df['decision']
control_rate = experiments_results_summary_df['control_rate'] * 100
treatment_rate = experiments_results_summary_df['treatment_rate'] * 100
difference = experiments_results_summary_df['treatment_rate'] - experiments_results_summary_df['control_rate']
lower_ci = experiments_results_summary_df['lower_ci']
upper_ci = experiments_results_summary_df['upper_ci']
alpha_used = experiments_results_summary_df['alpha']
is_stat_significant = bool(experiments_results_summary_df['statistically_significant'])
practical_threshold_pp = experiments_results_summary_df['practical_threshold_pp']
threshold_configured = has_value(practical_threshold_pp)
decision_reason = experiments_results_summary_df['decision_reason']
validity_status = experiments_results_summary_df['validity_status']
validity_reason = experiments_results_summary_df['validity_reason']
srm_p_value = experiments_results_summary_df['srm_p_value']

# Executive Decision Banner: one explicit entry per canonical decision state.
# (css class, heading color, text color, icon, message)
DECISION_BANNERS = {
    GREENLIGHT: ("decision-greenlight", "#00e676", "#b9f6ca", "✅",
                 "Statistically significant, and even the lower end of the 95% CI clears the practical threshold. Roll out the change."),
    PROMISING: ("decision-promising", "#40c4ff", "#b3e5fc", "🔷",
                "A worthwhile improvement is plausible, but the evidence does not yet establish practical value. Do not launch on this result alone."),
    LOW_IMPACT: ("decision-neutral", "#ffab40", "#ffe0b2", "➖",
                 "The effect appears positive, but even the upper end of the 95% CI is below the practical threshold."),
    INCONCLUSIVE: ("decision-warning", "#ffd600", "#fff59d", "⏸️",
                   "Inconclusive under the prespecified analysis. This is not evidence of no effect. "
                   "Continue only if the original stopping rule allows it, or plan a follow-up."),
    STOP_REVERT: ("decision-danger", "#ff1744", "#ff8a80", "🔴",
                  "Statistically significant harm: the entire 95% CI is below zero. Stop and revert the change."),
    INVALID: ("decision-invalid", "#d500f9", "#ea80fc", "🚫",
              "The experiment failed a validity check. Investigate randomization/assignment instrumentation before interpreting the treatment effect."),
    CONFIG_REQUIRED: ("decision-config", "#cfd8dc", "#b0bec5", "⚙️",
                      "No practical threshold is configured for this experiment, so a launch decision cannot be made. "
                      "Statistical results are shown below. Set the threshold in data/experiment_policy.csv, chosen independently of the observed result."),
}

st.markdown("---")
if decision in DECISION_BANNERS:
    css_class, heading_color, text_color, icon, message = DECISION_BANNERS[decision]
    st.markdown(f"""
    <div class="decision-card {css_class}">
        <h3 style="color: {heading_color}; margin-top: 0;">{icon} {decision}</h3>
        <p style="color: {text_color}; margin: 0.5rem 0 0 0;">{message}</p>
        <p style="color: {text_color}; opacity: 0.8; font-size: 0.9rem; margin: 0.5rem 0 0 0;"><strong>Why:</strong> {decision_reason}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error(f"Unrecognized decision label '{decision}'. Run `python refresh_results.py` to regenerate the results table.")

st.markdown("---")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Results", "🔬 Sensitivity Analysis", "📊 Trends", "🧮 Power Calculator"])

with tab1:
    st.markdown("## Experiment Results")

    # Validity gate first: an invalid experiment's effect should not be interpreted.
    if validity_status != VALIDITY_PASS:
        st.error(
            f"🚫 **Validity check failed.** {validity_reason} "
            "Investigate randomization/assignment instrumentation before interpreting the treatment effect. "
            "The numbers below are shown for diagnosis only."
        )

    # Top metrics rows: statistical evidence, then the business threshold and validity.
    metric_cols = st.columns(5)
    metric_cols[0].metric("Observed Effect", f"{difference*100:+.2f} pp", delta="Treatment − control", delta_color="off")
    metric_cols[1].metric("Relative Lift", f"{lift:+.2f}%")
    metric_cols[2].metric("p-value", f"{p_value:.4g}", delta=f"z = {z_score:.3f}", delta_color="off")
    metric_cols[3].metric("Statistically Significant?", "Yes" if is_stat_significant else "No",
                          delta=f"p ≤ α = {alpha_used:g}" if is_stat_significant else f"p > α = {alpha_used:g}", delta_color="off")
    metric_cols[4].metric("95% CI", f"[{lower_ci*100:+.2f}, {upper_ci*100:+.2f}] pp")

    metric_cols2 = st.columns(3)
    metric_cols2[0].metric("Practical Threshold (illustrative demo policy)",
                           f"{practical_threshold_pp:.2f} pp" if threshold_configured else "Not configured",
                           delta="Minimum worthwhile effect", delta_color="off")
    metric_cols2[1].metric("Decision", decision)
    # Overall validity verdict (SRM + cross-variant assignment) with the SRM p-value alongside it.
    srm_detail = f" (SRM p = {srm_p_value:.3g})" if has_value(srm_p_value) else ""
    cross_variant_users = int(experiments_results_summary_df['cross_variant_users'])
    metric_cols2[2].metric("Validity",
                           f"{validity_status}{srm_detail}",
                           delta=(f"SRM α = {experiments_results_summary_df['srm_alpha']:g} · "
                                  f"{cross_variant_users} user(s) in both variants"),
                           delta_color="off")
    st.caption(
        "Statistically significant is not the same as worth shipping: the decision also requires the effect to clear "
        "the practical threshold. The thresholds in this demo are illustrative values set after the synthetic data "
        "existed, not preregistered ones. In a real experiment, choose the threshold before looking at final results."
    )

    st.markdown("### Conversion Rate Comparison")
    comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

    with comparison_col1:
        st.markdown("**Control Group**")
        st.markdown(f"<h2 style='color: #1f77b4; margin: 0;'>{control_rate:.2f}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 0.95rem; font-weight: 600; color: #1f77b4;'>n = {int(experiments_results_summary_df['control_size']):,} users</p>", unsafe_allow_html=True)

    with comparison_col2:
        st.markdown("**vs.**")
        st.markdown("<h3 style='margin-top: 2rem;'>→</h3>", unsafe_allow_html=True)

    with comparison_col3:
        st.markdown("**Treatment Group**")
        st.markdown(f"<h2 style='color: #ff7f0e; margin: 0;'>{treatment_rate:.2f}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 0.95rem; font-weight: 600; color: #ff7f0e;'>n = {int(experiments_results_summary_df['treatment_size']):,} users</p>", unsafe_allow_html=True)

    st.markdown("### Confidence Interval & Practical Threshold")
    fig = ci_plot(difference, lower_ci, upper_ci, p_value,
                  practical_threshold_pp if threshold_configured else None, alpha_used)
    st.plotly_chart(fig, use_container_width=True)
    if threshold_configured:
        st.caption(
            "Read the interval by its endpoints. Does it cross the grey 0 pp line? "
            "Does the whole interval clear the blue practical-threshold line?"
        )
    else:
        st.caption(
            "Read the interval by its endpoints: does it cross the grey 0 pp line? "
            "No practical-threshold line is drawn because no threshold is configured."
        )

    st.markdown("### Analysis")
    summary = experiments_results_summary_df['summary']
    recommendation = experiments_results_summary_df['recommendation']

    analysis_col1, analysis_col2 = st.columns(2, gap="large")

    with analysis_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0d1b2a 0%, #1a2a3a 100%); border-left: 5px solid #00d4ff; border-radius: 8px; padding: 1.5rem; margin: 1rem 0;">
            <h4 style="color: #00d4ff; margin-top: 0; margin-bottom: 1rem;">📍 What Happened</h4>
            <p style="color: #cfd8dc; line-height: 1.6; margin: 0;">""" + summary + """</p>
        </div>
        """, unsafe_allow_html=True)

    with analysis_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a0d2a 0%, #2a1a3a 100%); border-left: 5px solid #7c4dff; border-radius: 8px; padding: 1.5rem; margin: 1rem 0;">
            <h4 style="color: #7c4dff; margin-top: 0; margin-bottom: 1rem;">🎯 Next Steps</h4>
            <p style="color: #cfd8dc; line-height: 1.6; margin: 0;">""" + recommendation + """</p>
        </div>
        """, unsafe_allow_html=True)

    # with st.expander("ℹ️ Assignment validity (SRM) details"):
    #     st.markdown(
    #         f"""
    #         Sample ratio mismatch (SRM) compares the users actually assigned to each variant with the planned split,
    #         using `data/experiment_assignments.csv`.

    #         | | Control | Treatment |
    #         |---|---|---|
    #         | Assigned users | {int(experiments_results_summary_df['assigned_control_n']):,} | {int(experiments_results_summary_df['assigned_treatment_n']):,} |
    #         | Expected users | {experiments_results_summary_df['expected_control_n']:,.1f} | {experiments_results_summary_df['expected_treatment_n']:,.1f} |

    #         - Planned treatment share: {experiments_results_summary_df['expected_treatment_share']:g}
    #         - Chi-square SRM p-value: {srm_p_value:.4g} (fails below SRM α = {experiments_results_summary_df['srm_alpha']:g})
    #         - Users assigned to both variants: {int(experiments_results_summary_df['cross_variant_users'])}
    #         """
    #     )

    # with st.expander("ℹ️ Analysis notes: peeking, multiple testing and guardrails"):
    #     st.markdown(
    #         """
    #         - **Optional stopping.** Standard p-values in this project assume a prespecified/fixed analysis plan.
    #           Repeatedly peeking and stopping when significance first appears can inflate Type I error. If continuous
    #           monitoring with arbitrary stopping is required, use a sequential testing design rather than this
    #           fixed-horizon decision rule.
    #         - **Multiple testing.** Each experiment is analysed on one prespecified primary metric, so no correction is
    #           applied. Secondary metrics or segment cuts would need a documented correction (for example Holm) or should
    #           be treated as exploratory. No multiple-testing correction is implemented in this version.
    #         - **Guardrails.** Production experiments often pair the primary metric with prespecified guardrail metrics
    #           (latency, refunds, errors). Guardrails are **not configured or evaluated** in this version.
    #         """
    #     )

with tab2:
    st.markdown("## Experiment Sensitivity Analysis")
    st.caption("Can this test reliably detect meaningful changes?")

    control_size = experiments_results_summary_df['control_size']
    treatment_size = experiments_results_summary_df['treatment_size']
    power = experiments_results_summary_df['power']
    mde_pp = experiments_results_summary_df['mde_pp']
    observed_pp = (experiments_results_summary_df['treatment_rate'] - experiments_results_summary_df['control_rate']) * 100
    required_users = experiments_results_summary_df['required_users']
    multiplier = experiments_results_summary_df['multiplier']
    total_required = experiments_results_summary_df['total_required']

    sens_col1, sens_col2, sens_col3, sens_col4 = st.columns(4)

    with sens_col1:
        st.metric("Power at Observed Effect Size", f"{power*100:.1f}%",
                  delta="Post-hoc; not evidence of reliability", delta_color="off")
    with sens_col2:
        st.metric("MDE at 80% Power", f"{mde_pp:.2f} pp", delta="Design sensitivity (α = 0.05)", delta_color="off")
    with sens_col3:
        st.metric("Practical Threshold (illustrative demo policy)",
                  f"{practical_threshold_pp:.2f} pp" if threshold_configured else "Not configured",
                  delta="Smallest change worth shipping", delta_color="off")
    with sens_col4:
        st.metric("Sample Size", f"{int(control_size):,} / {int(treatment_size):,}", delta="Per group")

    st.markdown("### Sample Size Requirement")

    req_col1, req_col2, req_col3 = st.columns(3)

    with req_col1:
        st.metric(
            "Users Required / Group",
            f"{int(required_users):,}",
            delta="For 80% power at observed effect",
            delta_color="off"
        )

    with req_col2:
        st.metric(
            "Total Users Required",
            f"{int(total_required):,}",
            delta="Control + treatment",
            delta_color="off"
        )

    with req_col3:
        st.metric(
            "Scale-Up Required",
            f"{multiplier:,.1f}×",
            delta="Relative to current control size",
            delta_color="off"
        )

    if multiplier > 1:
        st.warning(
            f"⚠️ At the observed effect size, this experiment would require approximately "
            f"**{multiplier:,.1f}× the current per-group sample size** to reach 80% power "
            f"— about **{int(required_users):,} users per group**."
        )
    else:
        st.success(
            "✅ The current sample size is sufficient to achieve approximately "
            "80% power for an effect of the observed magnitude."
        )

    st.caption(
        "MDE is a study-design quantity: what this test was powered to detect. The practical threshold is a business "
        "quantity: what effect is worth acting on. They answer different questions, and MDE is not used as the launch threshold."
    )

    st.markdown("### MDE vs Practical Threshold")
    if not threshold_configured:
        st.info(
            "⚙️ No practical threshold is configured, so the design sensitivity cannot be compared with the smallest "
            "worthwhile effect. Set it in data/experiment_policy.csv, independently of the observed result."
        )
    elif mde_pp > practical_threshold_pp:
        planning_warning = (
            f"⚠️ **Planning warning:** MDE ({mde_pp:.2f} pp) is larger than the practical threshold "
            f"({practical_threshold_pp:.2f} pp). This experiment's current design is less sensitive than the smallest "
            "effect the decision policy considers worthwhile. A practically important effect may be missed. "
            "This is a design/sensitivity warning, not a failure of the observed result."
        )
        if decision == GREENLIGHT:
            planning_warning += (
                "\n\nHere the risk did not materialise: the whole 95% CI already clears the practical threshold, so "
                "this warning does not affect the decision. Low design sensitivity does mean the observed effect may "
                "overstate the true effect, so expect a somewhat smaller lift after launch."
            )
        st.warning(planning_warning)
    else:
        st.success(
            f"✅ MDE ({mde_pp:.2f} pp) is at or below the practical threshold ({practical_threshold_pp:.2f} pp): "
            "the design was sensitive enough to detect the smallest worthwhile effect at 80% power."
        )

    st.markdown("---")

    st.markdown("### Power at Observed Effect Size")
    st.info(
        f"ℹ️ **{power*100:.0f}%** is the power this design would have if the true effect were exactly the observed "
        f"{observed_pp:+.2f} pp. Because it is computed from the observed effect, it rises and falls with the p-value: "
        "significant results always look well powered and non-significant results always look underpowered. "
        "It is therefore **not independent evidence** that the experiment was reliable or that a null result means "
        "no effect. To judge design sensitivity, compare the MDE with the practical threshold above."
    )

    st.markdown("### Effect Size Analysis")

    if abs(observed_pp) >= mde_pp:
        st.success(
            f"✅ The observed lift ({observed_pp:+.2f} pp) is **larger than the MDE** ({mde_pp:.2f} pp), "
            f"so an effect of this size was within the range the design had 80% power to detect."
        )
    else:
        st.warning(
            f"⚠️ The observed lift ({observed_pp:+.2f} pp) is **smaller than the MDE** ({mde_pp:.2f} pp). "
            f"The design had 80% power only for changes of {mde_pp:.2f} pp or larger, so smaller true effects "
            f"could easily go undetected."
        )

    st.markdown("---")
    st.markdown("### Key Insight")
    if is_stat_significant:
        insight = f"This experiment observed a {observed_pp:+.2f} pp change. The design had 80% power to detect effects of {mde_pp:.2f} pp or larger. The result was **statistically significant**, meaning the observed effect is unlikely to be due to chance alone. Whether it is worth shipping depends on the practical threshold, not on significance alone."
    else:
        insight = f"This experiment observed a {observed_pp:+.2f} pp change. The design had 80% power to detect effects of {mde_pp:.2f} pp or larger. The result was **not statistically significant**. Not significant does not mean no effect: true effects smaller than the MDE could easily have been missed."

    st.info(insight)

with tab3:
    st.markdown("## Conversion Rates Over Time")
    time_option = st.selectbox(
        "Choose a time period to view conversion rates",
        ["Weekly", "Monthly"],
        key="time_select"
    )

    if time_option == "Weekly":
        conversion_period = "conversion_rates_over_time_weekly.txt"
    else:
        conversion_period = "conversion_rates_over_time_monthly.txt"

    conversion_rates_control_df = get_df(conversion_period, (selected_experiment, 'control'))
    conversion_rates_treatment_df = get_df(conversion_period, (selected_experiment, 'treatment'))

    conversion_rates_df = pd.concat([conversion_rates_control_df, conversion_rates_treatment_df], ignore_index=True)

    conversion_rates_fig = px.line(
        conversion_rates_df,
        x="time_period",
        y="conversion_rate",
        color="variant",
        markers=True,
        title=f"{experiment} — Conversion Rate Trends ({time_option})",
        labels={"conversion_rate": "Conversion Rate (%)", "time_period": "Period", "variant": "Variant"},
        line_shape="spline",
        color_discrete_map={"control": "#1f77b4", "treatment": "#ff7f0e"}
    )

    conversion_rates_fig.update_layout(
        hovermode='x unified',
        height=450,
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(size=12, family="sans-serif", color='#cfd8dc'),
        title=dict(
            text=f"{experiment} — Conversion Rate Trends ({time_option})",
            font=dict(size=16, color='#00d4ff', family="sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title_font_size=13,
        yaxis_title_font_size=13,
        xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='#1e2a3a',
            showline=True, linewidth=1, linecolor='#2d3548',
            tickfont=dict(size=11, color='#90a4ae')
        ),
        yaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='#1e2a3a',
            showline=True, linewidth=1, linecolor='#2d3548',
            tickfont=dict(size=11, color='#90a4ae')
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='#1a1f2e',
            bordercolor='#2d3548',
            borderwidth=1,
            font=dict(size=12, color='#cfd8dc')
        )
    )

    conversion_rates_fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8, symbol='circle')
    )

    st.plotly_chart(conversion_rates_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Interpretation")
    st.markdown(
        """
        - **Control (blue line)**: Conversion rate for users in the control variant
        - **Treatment (orange line)**: Conversion rate for users in the treatment variant
        - **Visual Gap**: The distance between lines shows the magnitude of the effect
        - **Trend Stability**: Flat or consistent lines indicate stable, reliable results
        """
    )

# ============================================================================
# POWER CALCULATOR TAB - FIXED VERSION FOR charts.py
# ============================================================================
# 
# INSTRUCTIONS:
# 1. Find line 236 in charts.py where it says:
#    tab1, tab2, tab3 = st.tabs(["📈 Results", "🔬 Sensitivity Analysis", "📊 Trends"])
# 
# 2. REPLACE that line with:
#    tab1, tab2, tab3, tab4 = st.tabs(["📈 Results", "🔬 Sensitivity Analysis", "📊 Trends", "🧮 Power Calculator"])
#
# 3. Then ADD this entire section at the END of the file (after the tab3 section ends at line 421)
# ============================================================================

# Import the functions from statistical_tests.py at the top of charts.py
# Add this line after line 7 (after the other imports):
from statistical_tests import (
    calculate_statistical_power_from_results,
    calculate_sample_users_from_results,
    calculate_minimum_detectable_effect_from_results
)

# ============================================================================
# POWER CALCULATOR
# ============================================================================

with tab4:
    # Helper function to safely extract scalar values from various return types
    def safe_scalar(value):
        """Convert numpy arrays, pandas Series, or other iterables to Python scalars"""
        if hasattr(value, 'iloc'):
            # It's a pandas Series
            return safe_scalar(value.iloc[0])
        elif hasattr(value, 'item'):
            # It's a numpy scalar
            return value.item()
        elif hasattr(value, '__len__') and not isinstance(value, str):
            # It's array-like (but not a string)
            if len(value) > 0:
                return safe_scalar(value[0])
            else:
                return 0
        else:
            # It's already a scalar
            return value
    
    st.markdown("## 🧮 Statistical Power Calculator")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d1b2a 0%, #1a2a3a 100%); border-left: 5px solid #7c4dff; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem;">
        <h4 style="color: #7c4dff; margin-top: 0;">What is Statistical Power?</h4>
        <p style="color: #cfd8dc; line-height: 1.6; margin: 0.5rem 0;">
            <strong>Statistical Power</strong> is the probability that your test will detect a real difference when one exists. 
            Industry standard is <strong>80%</strong> (4 in 5 chance of detecting true effects).
        </p>
        <p style="color: #cfd8dc; line-height: 1.6; margin: 0.5rem 0 0 0;">
            Use this calculator to plan your experiments and avoid underpowered tests that might miss real improvements!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three calculator modes
    calc_mode = st.radio(
        "Choose Calculator Mode:",
        ["📏 Sample Size Calculator", "⚡ Power Analysis", "🎯 Minimum Detectable Effect"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # ========================================================================
    # MODE 1: SAMPLE SIZE CALCULATOR
    # ========================================================================
    if calc_mode == "📏 Sample Size Calculator":
        st.markdown("### Calculate Required Sample Size")
        st.caption("How many users do you need to detect a specific improvement?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            baseline_rate_pct = st.number_input(
                "Baseline Conversion Rate (%)",
                min_value=0.1,
                max_value=99.0,
                value=5.0,
                step=0.1,
                help="Your current conversion rate (before the experiment)"
            )
            baseline_rate = baseline_rate_pct / 100
            
            improvement_type = st.radio(
                "Specify improvement as:",
                ["Absolute (percentage points)", "Relative (% lift)"]
            )
            
            if improvement_type == "Absolute (percentage points)":
                improvement_abs = st.number_input(
                    "Expected Improvement (percentage points)",
                    min_value=0.1,
                    max_value=50.0,
                    value=1.0,
                    step=0.1,
                    help="e.g., 5% → 6% is a 1 percentage point improvement"
                )
                target_rate = baseline_rate + (improvement_abs / 100)
            else:
                improvement_rel = st.number_input(
                    "Expected Relative Lift (%)",
                    min_value=1.0,
                    max_value=500.0,
                    value=20.0,
                    step=1.0,
                    help="e.g., 20% lift on 5% baseline = 6% target (1% absolute)"
                )
                target_rate = baseline_rate * (1 + improvement_rel / 100)
        
        with col2:
            alpha = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                help="Probability of false positive (Type I error). Standard is 5%."
            )
            
            power = st.slider(
                "Desired Statistical Power",
                min_value=0.70,
                max_value=0.95,
                value=0.80,
                step=0.05,
                help="Probability of detecting a real effect. Standard is 80%."
            )
        
        if st.button("🔢 Calculate Sample Size", type="primary", use_container_width=True):
            try:
                # Create a mock summary DataFrame for the function
                mock_summary = pd.DataFrame({
                    'control_rate': [baseline_rate],
                    'treatment_rate': [target_rate],
                    'control_size': [10000],  # Dummy value, not used in calculation
                    'treatment_size': [10000]  # Dummy value, not used in calculation
                })
                
                result = calculate_sample_users_from_results(mock_summary, alpha=alpha, power=power)
                
                # Extract scalar values from the returned tuple
                required_users = int(safe_scalar(result[0]))
                total_required = int(safe_scalar(result[1]))
                multiplier = float(safe_scalar(result[2]))
                
                abs_improvement = (target_rate - baseline_rate) * 100
                rel_improvement = ((target_rate - baseline_rate) / baseline_rate) * 100
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                # Key metrics
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.markdown("""
                        <div style="background: #1a1f2e; border: 2px solid #00d4ff; border-radius: 8px; padding: 1.5rem; text-align: center;">
                            <p style="color: #90a4ae; margin: 0; font-size: 0.9rem;">Users Per Group</p>
                            <h2 style="color: #00d4ff; margin: 0.5rem 0;">{:,}</h2>
                        </div>
                    """.format(required_users), unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown("""
                        <div style="background: #1a1f2e; border: 2px solid #7c4dff; border-radius: 8px; padding: 1.5rem; text-align: center;">
                            <p style="color: #90a4ae; margin: 0; font-size: 0.9rem;">Total Experiment Size</p>
                            <h2 style="color: #7c4dff; margin: 0.5rem 0;">{:,}</h2>
                        </div>
                    """.format(total_required), unsafe_allow_html=True)
                
                with result_col3:
                    st.markdown("""
                        <div style="background: #1a1f2e; border: 2px solid #00e676; border-radius: 8px; padding: 1.5rem; text-align: center;">
                            <p style="color: #90a4ae; margin: 0; font-size: 0.9rem;">Detecting</p>
                            <h2 style="color: #00e676; margin: 0.5rem 0;">{:.2f}%</h2>
                            <p style="color: #90a4ae; margin: 0; font-size: 0.8rem;">relative lift</p>
                        </div>
                    """.format(rel_improvement), unsafe_allow_html=True)
                
                # Interpretation
                st.markdown("### 💡 Interpretation")
                st.info(f"""
                **To detect a {abs_improvement:.2f} percentage point improvement** (from {baseline_rate_pct:.1f}% to {target_rate*100:.1f}%):
                
                - **Need {required_users:,} users per group** (Control and Treatment)
                - **Total experiment size: {total_required:,} users**
                - **This gives you {power*100:.0f}% power** (probability of detecting this effect if it exists)
                - **With {alpha*100:.0f}% false positive rate** (probability of incorrectly detecting an effect)
                
                This is a **{rel_improvement:.1f}% relative lift** on your baseline conversion rate.
                """)
                
                # Timeline estimation
                st.markdown("### ⏱️ Timeline Estimation")
                daily_traffic = st.number_input(
                    "Daily Traffic (users per day)",
                    min_value=10,
                    max_value=1000000,
                    value=1000,
                    step=100
                )
                
                days_needed = total_required / daily_traffic
                weeks_needed = days_needed / 7
                
                timeline_col1, timeline_col2 = st.columns(2)
                with timeline_col1:
                    st.metric("Days Needed", f"{days_needed:.1f}")
                with timeline_col2:
                    st.metric("Weeks Needed", f"{weeks_needed:.1f}")
                
                if days_needed < 7:
                    st.success(f"✅ Short experiment! Can be completed in under a week.")
                elif days_needed < 30:
                    st.info(f"📅 Reasonable timeline: {weeks_needed:.1f} weeks")
                else:
                    st.warning(f"⏳ Long experiment: {weeks_needed:.1f} weeks. Consider testing larger changes or increasing traffic.")
                
            except Exception as e:
                st.error(f"Error calculating sample size: {str(e)}")
    
    # ========================================================================
    # MODE 2: POWER ANALYSIS
    # ========================================================================
    elif calc_mode == "⚡ Power Analysis":
        st.markdown("### Calculate Statistical Power")
        st.caption("With your current sample size, what's your probability of detecting an effect?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size_per_group = st.number_input(
                "Sample Size Per Group",
                min_value=10,
                max_value=1000000,
                value=2000,
                step=100,
                help="How many users you have (or plan to have) in each group"
            )
            
            baseline_rate_pct = st.number_input(
                "Baseline Conversion Rate (%)",
                min_value=0.1,
                max_value=99.0,
                value=5.0,
                step=0.1,
                key="power_baseline"
            )
            baseline_rate = baseline_rate_pct / 100
        
        with col2:
            target_rate_pct = st.number_input(
                "Target Conversion Rate (%)",
                min_value=0.1,
                max_value=99.0,
                value=6.0,
                step=0.1,
                help="The improvement you want to detect"
            )
            target_rate = target_rate_pct / 100
            
            alpha = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                key="power_alpha"
            )
        
        if st.button("⚡ Calculate Power", type="primary", use_container_width=True):
            try:
                # Create a mock summary DataFrame for the function
                mock_summary = pd.DataFrame({
                    'control_rate': [baseline_rate],
                    'treatment_rate': [target_rate],
                    'control_size': [sample_size_per_group],
                    'treatment_size': [sample_size_per_group]
                })
                
                achieved_power_raw = calculate_statistical_power_from_results(mock_summary, alpha=alpha)
                
                # Extract scalar value if it's a numpy array or pandas Series
                achieved_power = float(safe_scalar(achieved_power_raw))
                
                abs_improvement = (target_rate - baseline_rate) * 100
                rel_improvement = ((target_rate - baseline_rate) / baseline_rate) * 100
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                # Power gauge
                power_pct = achieved_power * 100
                
                if achieved_power >= 0.8:
                    power_color = "#00e676"
                    power_status = "High Power ✅"
                    power_message = "Excellent! Very likely to detect this effect if it exists."
                elif achieved_power >= 0.6:
                    power_color = "#ffd600"
                    power_status = "Moderate Power ⚠️"
                    power_message = "Decent, but consider increasing sample size for more reliability."
                else:
                    power_color = "#ff1744"
                    power_status = "Low Power ❌"
                    power_message = "Underpowered! High risk of missing real effects."
                
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #0d1b2a 0%, #1a2a3a 100%); border-left: 5px solid {power_color}; border-radius: 8px; padding: 2rem; margin: 1rem 0;">
                        <h2 style="color: {power_color}; margin: 0 0 1rem 0;">{power_pct:.1f}%</h2>
                        <h4 style="color: {power_color}; margin: 0 0 1rem 0;">{power_status}</h4>
                        <p style="color: #cfd8dc; margin: 0;">{power_message}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detailed metrics
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric("Sample Size", f"{sample_size_per_group:,} per group")
                with result_col2:
                    st.metric("Effect to Detect", f"{abs_improvement:.2f} pp")
                with result_col3:
                    st.metric("Relative Lift", f"{rel_improvement:.1f}%")
                
                # Interpretation
                st.markdown("### 💡 Interpretation")
                st.info(f"""
                With **{sample_size_per_group:,} users per group**, you have:
                
                - **{power_pct:.1f}% probability** of detecting a {abs_improvement:.2f} pp improvement
                - **{(1-achieved_power)*100:.1f}% risk** of missing this effect (Type II error)
                - This assumes a true difference of {baseline_rate_pct:.1f}% → {target_rate_pct:.1f}%
                
                **What does this mean?**
                If the treatment is truly {rel_improvement:.1f}% better, you'll detect it in {int(achieved_power*100)} out of 100 experiments.
                """)
                
                # Recommendation
                if achieved_power < 0.8:
                    # Calculate needed sample size for 80% power
                    mock_summary_80 = pd.DataFrame({
                        'control_rate': [baseline_rate],
                        'treatment_rate': [target_rate],
                        'control_size': [10000],
                        'treatment_size': [10000]
                    })
                    result_80 = calculate_sample_users_from_results(mock_summary_80, alpha=alpha, power=0.80)
                    needed_users = int(safe_scalar(result_80[0]))
                    additional_users = needed_users - sample_size_per_group
                    
                    st.warning(f"""
                    ⚠️ **Recommendation:** To achieve 80% power (industry standard) at α = {alpha:.2f}, you need:
                    - **{needed_users:,} users per group** 
                    - **{additional_users:,} more users** than you currently have
                    - Or test for a larger effect size
                    """)
                
            except Exception as e:
                st.error(f"Error calculating power: {str(e)}")
    
    # ========================================================================
    # MODE 3: MINIMUM DETECTABLE EFFECT
    # ========================================================================
    else:  # Minimum Detectable Effect
        st.markdown("### Calculate Minimum Detectable Effect")
        st.caption("What's the smallest improvement you can reliably detect with your sample size?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size_per_group = st.number_input(
                "Sample Size Per Group",
                min_value=10,
                max_value=1000000,
                value=2000,
                step=100,
                help="How many users you have in each group",
                key="mde_sample"
            )
            
            baseline_rate_pct = st.number_input(
                "Baseline Conversion Rate (%)",
                min_value=0.1,
                max_value=99.0,
                value=5.0,
                step=0.1,
                key="mde_baseline"
            )
            baseline_rate = baseline_rate_pct / 100
        
        with col2:
            alpha = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                key="mde_alpha"
            )
            
            power = st.slider(
                "Desired Statistical Power",
                min_value=0.70,
                max_value=0.95,
                value=0.80,
                step=0.05,
                key="mde_power"
            )
        
        if st.button("🎯 Calculate MDE", type="primary", use_container_width=True):
            try:
                # Create a mock summary DataFrame for the function
                mock_summary = pd.DataFrame({
                    'control_rate': [baseline_rate],
                    'treatment_rate': [baseline_rate],  # Will be solved
                    'control_size': [sample_size_per_group],
                    'treatment_size': [sample_size_per_group],
                    'lift_percent': [0]  # Placeholder
                })
                
                mde_dict = calculate_minimum_detectable_effect_from_results(mock_summary, alpha=alpha, power=power)
                
                mde_pp = float(safe_scalar(mde_dict['mde_pp']))
                mde_relative = float(safe_scalar(mde_dict['mde_relative_lift_pct']))
                target_rate = baseline_rate + (mde_pp / 100)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Results")
                
                # Key metrics
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.markdown(f"""
                        <div style="background: #1a1f2e; border: 2px solid #00d4ff; border-radius: 8px; padding: 1.5rem; text-align: center;">
                            <p style="color: #90a4ae; margin: 0; font-size: 0.9rem;">MDE (Absolute)</p>
                            <h2 style="color: #00d4ff; margin: 0.5rem 0;">{mde_pp:.2f} pp</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown(f"""
                        <div style="background: #1a1f2e; border: 2px solid #7c4dff; border-radius: 8px; padding: 1.5rem; text-align: center;">
                            <p style="color: #90a4ae; margin: 0; font-size: 0.9rem;">MDE (Relative)</p>
                            <h2 style="color: #7c4dff; margin: 0.5rem 0;">{mde_relative:.1f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                with result_col3:
                    st.markdown(f"""
                        <div style="background: #1a1f2e; border: 2px solid #00e676; border-radius: 8px; padding: 1.5rem; text-align: center;">
                            <p style="color: #90a4ae; margin: 0; font-size: 0.9rem;">Target Rate</p>
                            <h2 style="color: #00e676; margin: 0.5rem 0;">{target_rate*100:.2f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Interpretation
                st.markdown("### 💡 Interpretation")
                st.info(f"""
                With **{sample_size_per_group:,} users per group** and **{power*100:.0f}% power**:
                
                - **Smallest detectable improvement: {mde_pp:.2f} percentage points**
                - This is a **{mde_relative:.1f}% relative lift** on your {baseline_rate_pct:.1f}% baseline
                - Can reliably detect changes from {baseline_rate_pct:.1f}% → {target_rate*100:.2f}% or larger
                - Smaller improvements might be missed
                
                **What does this mean?**
                Effects smaller than {mde_pp:.2f} pp are likely to go undetected with your current sample size.
                """)
                
                # Practical guidance
                st.markdown("### 🎯 Practical Guidance")
                
                if mde_relative > 30:
                    st.warning(f"""
                    ⚠️ **Large MDE ({mde_relative:.1f}%)** — You can only detect very large improvements.
                    
                    **Recommendations:**
                    - Increase sample size for more sensitivity
                    - Focus on testing major changes, not minor optimizations
                    - Consider A/B tests only for significant redesigns
                    """)
                elif mde_relative > 15:
                    st.info(f"""
                    📊 **Moderate MDE ({mde_relative:.1f}%)** — Good for testing meaningful changes.
                    
                    **Recommendations:**
                    - Suitable for testing significant feature changes
                    - May miss small micro-optimizations
                    - Consider if {mde_relative:.1f}% lift is worth the effort
                    """)
                else:
                    st.success(f"""
                    ✅ **Small MDE ({mde_relative:.1f}%)** — Highly sensitive test!
                    
                    **Recommendations:**
                    - Can detect even small improvements
                    - Good for fine-tuning and optimization
                    - Be cautious of statistical vs. practical significance
                    """)
                
            except Exception as e:
                st.error(f"Error calculating MDE: {str(e)}")
    
    # ========================================================================
    # HELPFUL RESOURCES SECTION
    # ========================================================================
    st.markdown("---")
    st.markdown("### 📚 Understanding the Concepts")
    
    with st.expander("ℹ️ What is Statistical Power?"):
        st.markdown("""
        **Statistical Power** is the probability that your test will correctly detect a real effect when one exists.
        
        **Example:**
        - If power = 80%, and there IS a real 10% improvement...
        - You'll detect it 80 out of 100 times
        - You'll miss it 20 out of 100 times (Type II error)
        
        **Industry Standard:** 80% (sometimes 90% for critical decisions)
        
        **Low power is bad because:**
        - You might conclude "no difference" when one actually exists
        - Wasted experiment (collected data but couldn't detect the effect)
        - Missed opportunity for improvement
        """)
    
    with st.expander("ℹ️ What affects Statistical Power?"):
        st.markdown("""
        Four factors determine power:
        
        1. **Sample Size** ⬆️ = Power ⬆️
           - More data = easier to detect differences
           - Doubling sample size doesn't double power
        
        2. **Effect Size** ⬆️ = Power ⬆️
           - Larger improvements are easier to detect
           - 20% lift is easier to detect than 2% lift
        
        3. **Significance Level (α)** ⬆️ = Power ⬆️
           - But this increases false positives!
           - Standard is 5% (α = 0.05)
        
        4. **Variance/Noise** ⬆️ = Power ⬇️
           - Less variability = clearer signal
           - Binary outcomes (conversion) have fixed variance
        """)
    
    with st.expander("ℹ️ What is Minimum Detectable Effect (MDE)?"):
        st.markdown("""
        **MDE** is the smallest improvement you can reliably detect given:
        - Your sample size
        - Desired power (usually 80%)
        - Significance level (usually 5%)
        
        **Example:**
        - Baseline: 5% conversion
        - Sample: 2,000 per group
        - MDE: 1.2 percentage points
        - Meaning: Can detect 5% → 6.2% (or larger)
        - Cannot reliably detect 5% → 5.5%
        
        **Use MDE to:**
        - Set realistic expectations before testing
        - Decide if your sample size is adequate
        - Determine if an experiment is worth running
        """)
    
    with st.expander("🎯 Best Practices"):
        st.markdown("""
        **Before Running an Experiment:**
        1. ✅ Calculate required sample size for desired effect
        2. ✅ Ensure you have enough traffic/time
        3. ✅ Pre-register your hypothesis and sample size
        
        **During the Experiment:**
        1. ❌ Don't peek and stop early (inflates false positives)
        2. ✅ Run for full duration or sample size
        3. ✅ Account for weekly/seasonal patterns
        
        **After the Experiment:**
        1. ✅ Check if you had sufficient power
        2. ✅ "Not significant" ≠ "No difference" (might be underpowered!)
        3. ✅ Consider practical significance vs statistical significance
        
        **Power Guidelines:**
        - Minimum: 70% (acceptable for exploratory tests)
        - Standard: 80% (industry norm)
        - Conservative: 90% (for critical decisions)
        """)