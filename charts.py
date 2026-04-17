from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statistical_tests import calculate_statistical_power_from_results,calculate_sample_users_from_results,calculate_minimum_detectable_effect_from_results

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
tab1, tab2, tab3, tab4 = st.tabs(["📈 Results", "🔬 Sensitivity Analysis", "📊 Trends", "🧮 Power Calculator"])

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
                
                result = calculate_sample_users_from_results(mock_summary)
                
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
                
                achieved_power_raw = calculate_statistical_power_from_results(mock_summary)
                
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
                    result_80 = calculate_sample_users_from_results(mock_summary_80)
                    needed_users = int(safe_scalar(result_80[0]))
                    additional_users = needed_users - sample_size_per_group
                    
                    st.warning(f"""
                    ⚠️ **Recommendation:** To achieve 80% power (industry standard), you need:
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
                
                mde_dict = calculate_minimum_detectable_effect_from_results(mock_summary)
                
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