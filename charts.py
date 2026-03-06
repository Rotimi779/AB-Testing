from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

#For confidence interval plots
def ci_plot(difference, lower_ci, upper_ci, p_value):
    diff_pp  = difference * 100
    low_pp   = lower_ci * 100
    high_pp  = upper_ci * 100

    significant = (low_pp > 0) or (high_pp < 0)
    color = "seagreen" if significant else "darkorange"
    status = "Significant" if significant else "Not significant"

    xmin = min(low_pp, 0, diff_pp)
    xmax = max(high_pp, 0, diff_pp)
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
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

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
    'Free Shipping Threshold': 5
}

with col_selector:
    experiment = st.selectbox(
        'Select Experiment',
        ['Checkout Button Color', 'Pricing Display Test', 'Email Subject Line', 'Product Page Layout', 'Free Shipping Threshold'],
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

# Executive Decision Banner
st.markdown("---")
if decision == "GREENLIGHT":
    st.markdown("""
    <div class="decision-card decision-greenlight">
        <h3 style="color: #00e676; margin-top: 0;">✅ GREENLIGHT</h3>
        <p style="color: #b9f6ca; margin: 0.5rem 0 0 0;">This experiment shows statistically significant improvement. Roll out the change.</p>
    </div>
    """, unsafe_allow_html=True)
elif decision == "KEEP RUNNING":
    st.markdown("""
    <div class="decision-card decision-warning">
        <h3 style="color: #ffd600; margin-top: 0;">🟨 KEEP RUNNING</h3>
        <p style="color: #fff59d; margin: 0.5rem 0 0 0;">Results are inconclusive. Continue the experiment to gather more data or increase sample size.</p>
    </div>
    """, unsafe_allow_html=True)
elif decision == "STOP / REVERT":
    st.markdown("""
    <div class="decision-card decision-danger">
        <h3 style="color: #ff1744; margin-top: 0;">🔴 STOP / REVERT</h3>
        <p style="color: #ff8a80; margin: 0.5rem 0 0 0;">This experiment shows statistically significant degradation. Stop and revert the change.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="decision-card decision-warning">
        <h3 style="color: #ffd600; margin-top: 0;">ℹ️ LOW IMPACT</h3>
        <p style="color: #fff59d; margin: 0.5rem 0 0 0;">Result is significant but effect is small. Evaluate business value vs. implementation cost.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📈 Results", "🔬 Sensitivity Analysis", "📊 Trends"])

with tab1:
    st.markdown("## Experiment Results")

    # Top metrics row
    metric_cols = st.columns(4)
    metric_cols[0].metric("Observed Lift", f"{lift:+.2f}%", delta=f"{difference*100:+.2f} pp")
    metric_cols[1].metric("p-value", f"{p_value:.4g}", delta="Significance test")
    metric_cols[2].metric("z-score", f"{z_score:.3f}", delta="Test statistic")
    metric_cols[3].metric("Significant?", "Yes ✅" if experiments_results_summary_df["is_significant"] else "No ❌")

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

    st.markdown("### Confidence Interval & Statistical Significance")
    fig = ci_plot(difference, lower_ci, upper_ci, p_value)
    st.plotly_chart(fig, use_container_width=True)

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

with tab2:
    st.markdown("## Experiment Sensitivity Analysis")
    st.caption("Can this test reliably detect meaningful changes?")

    control_size = experiments_results_summary_df['control_size']
    treatment_size = experiments_results_summary_df['treatment_size']
    power = experiments_results_summary_df['power']
    mde_pp = experiments_results_summary_df['mde_pp']
    observed_pp = (experiments_results_summary_df['treatment_rate'] - experiments_results_summary_df['control_rate']) * 100

    sens_col1, sens_col2, sens_col3 = st.columns(3)

    with sens_col1:
        st.metric("Statistical Power", f"{power*100:.1f}%", delta="Probability of detecting true effect")
    with sens_col2:
        st.metric("Minimum Detectable Effect", f"{mde_pp:.2f} pp", delta="Smallest reliably detectable change")
    with sens_col3:
        st.metric("Sample Size", f"{int(control_size):,} / {int(treatment_size):,}", delta="Per group")

    st.markdown("---")

    st.markdown("### Statistical Power Interpretation")
    if power >= 0.8:
        st.success(f"✅ **High Power ({power*100:.0f}%)** — This experiment was very likely to detect a real effect if one existed.")
    elif power >= 0.6:
        st.warning(f"🟨 **Moderate Power ({power*100:.0f}%)** — Some chance of detecting the effect, but not guaranteed.")
    else:
        st.error(f"❌ **Low Power ({power*100:.0f}%)** — This experiment may have missed real effects.")

    st.markdown("### Effect Size Analysis")

    if abs(observed_pp) >= mde_pp:
        st.success(
            f"✅ The observed lift ({observed_pp:+.2f} pp) is **larger than the MDE** ({mde_pp:.2f} pp). "
            f"The test was sensitive enough to detect this effect reliably."
        )
    else:
        st.warning(
            f"⚠️ The observed lift ({observed_pp:+.2f} pp) is **smaller than the MDE** ({mde_pp:.2f} pp). "
            f"The test could only reliably detect changes of {mde_pp:.2f} pp or larger."
        )

    st.markdown("---")
    st.markdown("### Key Insight")
    if experiments_results_summary_df["is_significant"]:
        insight = f"This experiment observed a {observed_pp:+.2f} pp change with {power*100:.0f}% power to detect effects as small as {mde_pp:.2f} pp. The result was **statistically significant**, meaning the observed effect is unlikely to be due to chance."
    else:
        insight = f"This experiment observed a {observed_pp:+.2f} pp change with {power*100:.0f}% power to detect effects as small as {mde_pp:.2f} pp. The result was **not statistically significant**, likely because the effect size was smaller than the MDE or the test lacked sufficient power."

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