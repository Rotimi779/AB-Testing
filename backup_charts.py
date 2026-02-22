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
def ci_plot(difference, lower_ci, upper_ci,p_value):
    # Convert to percentage points for display
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
        marker=dict(size=12,color=color),
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
            +"<extra></extra>"
        )
    ))

    # Reference line at 0 (no effect)
    fig.add_vline(x=0, line_dash="dash", line_color ="gray")


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
        bgcolor="rgba(0,0,0,0.4)",
        bordercolor=color,
        borderwidth=1
    )

    fig.update_layout(
        xaxis_title="Percentage points (pp)",
        yaxis_title="",
        showlegend=False,
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_xaxes(range=x_range, zeroline=False)
    return fig

st.set_page_config(page_title="A/B Testing Analysis", layout="wide")
st.title("A/B Testing Dashboard")

st.header("Experiment Analysis")
st.info(
    """
    **Experiment Overview**

    This section summarizes the performance of ongoing A/B experiments by comparing
    control and treatment variants across key metrics. For each experiment, we report **statistical test results** including lift percentage,
    z-scores, and p-values, visualize **confidence intervals** around
    observed lifts, and analyze
    **user-based conversion rates over time**.

    These views together help assess both the **magnitude** and **statistical reliability**
    of observed differences, enabling informed decisions on whether an experiment shows
    meaningful impact.
    """
)

st.subheader("Statistical test results")

experiment_mapping_dict = {'Checkout Button Color':1, 'Pricing Display Test':2, 'Email Subject Line': 3, 'Product Page Layout': 4, 'Free Shipping Threshold': 5}


experiment = st.selectbox('Select an experiment',['Checkout Button Color', 'Pricing Display Test', 'Email Subject Line', 'Product Page Layout', 'Free Shipping Threshold'])
selected_experiment = experiment_mapping_dict[experiment]


experiments_results_summary = 'experiments_results_summary.txt'
experiments_results_summary_df = get_df(experiments_results_summary,(selected_experiment,)).iloc[0]
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Lift", f"{experiments_results_summary_df['lift_percent']:+.2f}%" if experiments_results_summary_df["lift_percent"] is not None else "N/A")
metric2.metric("p-value", f"{experiments_results_summary_df['p_value']:.4g}")
metric3.metric("z-score", f"{experiments_results_summary_df['z_score']:.3f}")
#Change metric here
metric4.metric("Significant?", "Yes ✅" if experiments_results_summary_df["is_significant"] else "No ❌")



##CONFIDENCE INTERVAL PLOTS SHOULD BE RIGHT HERE!!!!!!!!
st.subheader("Confidence Interval")
difference = experiments_results_summary_df['treatment_rate'] - experiments_results_summary_df['control_rate']
fig = ci_plot(difference, experiments_results_summary_df["lower_ci"], experiments_results_summary_df["upper_ci"],experiments_results_summary_df["p_value"])
st.plotly_chart(fig, use_container_width=True)


control_rate = experiments_results_summary_df['control_rate'] * 100
treatment_rate = experiments_results_summary_df['treatment_rate'] * 100
lift = experiments_results_summary_df['lift_percent']

metric_types = {'conversion': 'conversion', 'revenue': 'revenue', 'click_through': 'click through', 'engagement': 'engagement'}

metric_to_track = metric_types[experiments_results_summary_df['metric_type']]

p_value = experiments_results_summary_df["p_value"]
#Lower limit and upper limit for these experiments to see just how close their p-values were to 0.05
# lower_limit_to_experiment = 0.05 * 0.8
# upper_limit_to_experiment = 0.05 * 1.2


lower_ci = experiments_results_summary_df['lower_ci']
upper_ci = experiments_results_summary_df['upper_ci']

z_score = round(experiments_results_summary_df['z_score'],4)

summary = experiments_results_summary_df['summary']
recommendation = experiments_results_summary_df['recommendation']
decision = experiments_results_summary_df['decision']


st.info(
    f"""
        **Experiment**: {experiment}\
        \n**Hypothesis**: {experiments_results_summary_df['hypothesis']}, testing for {metric_to_track}.\
        \n\nFor this experiment, the initial conversion rate for the controlled scenario was **{round(control_rate,2)}**% and the conversion rate for the treatment
        scenario was **{round(treatment_rate,2)}**%. This experiment experienced a lift of {lift}%.\
    """
)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Summary")
    st.write(summary)
with col2:
    st.markdown("### Recommendation")
    st.write(recommendation)
with col3:
    st.markdown("### Final Decision")
    if decision == "GREENLIGHT":
        st.success(f"✅ Greenlight the change")
    elif decision == "KEEP RUNNING":
        st.warning(f"🟨 Keep running the experiment")
    elif decision == "STOP / REVERT":
        st.error(f"🟥 Stop and revert the change")
    else:
        st.info(f"ℹ️ Low impact")
    
st.divider()

st.markdown("### Experiment Sensitivity")
st.caption("How reliable was this test? Could it detect meaningful changes?")

# Get sensitivity metrics
control_size = experiments_results_summary_df['control_size']
treatment_size = experiments_results_summary_df['treatment_size']
power = experiments_results_summary_df['power']
mde_pp = experiments_results_summary_df['mde_pp']
observed_pp = (experiments_results_summary_df['treatment_rate'] - experiments_results_summary_df['control_rate']) * 100

# Display metrics
sens_col1, sens_col2, sens_col3 = st.columns(3)
sens_col1.metric("Sample Size (per group)", f"{int(control_size):,} / {int(treatment_size):,}")
sens_col2.metric("Minimum Detectable Effect", f"{mde_pp:.2f} pp")
sens_col3.metric("Statistical Power", f"{power*100:.1f}%")

# Interpretation
st.markdown("#### What this means:")

# Power interpretation
if power >= 0.8:
    power_text = f"**High power ({power*100:.0f}%):** This experiment was very likely to detect a real effect if one existed."
elif power >= 0.6:
    power_text = f"**Moderate power ({power*100:.0f}%):** This experiment had some chance of detecting the effect, but not guaranteed."
else:
    power_text = f"**Low power ({power*100:.0f}%):** This experiment may have missed real effects."

# MDE and observed comparison
if abs(observed_pp) >= mde_pp:
    mde_comparison = f"✅ The observed lift ({observed_pp:+.2f} pp) is **larger than the MDE** ({mde_pp:.2f} pp). The test was sensitive enough to detect this effect."
else:
    mde_comparison = f"⚠️ The observed lift ({observed_pp:+.2f} pp) is **smaller than the MDE** ({mde_pp:.2f} pp). The test could only reliably detect changes of {mde_pp:.2f} pp or larger."

st.write(f"{power_text}\n\n{mde_comparison}")

st.divider()

# Final experiment summary
st.markdown("#### Experiment Summary")
if experiments_results_summary_df["is_significant"]:
    summary_text = f"This experiment observed a {observed_pp:+.2f} pp change with {power*100:.0f}% power to detect effects as small as {mde_pp:.2f} pp. **The result was statistically significant.**"
else:
    summary_text = f"This experiment observed a {observed_pp:+.2f} pp change with {power*100:.0f}% power to detect effects as small as {mde_pp:.2f} pp. The result was not statistically significant, likely because the effect size was smaller than the MDE."

st.info(summary_text)

st.divider()

st.subheader("Conversion Rates over time")
time_option = st.selectbox(
        "Choose a time period to view conversion rates",
        ["Weekly", "Monthly"]   
    )



conversion_period = "conversion_rates_over_time.txt"
if time_option == "Weekly":
    conversion_period = "conversion_rates_over_time_weekly.txt"
else:
    conversion_period = "conversion_rates_over_time_monthly.txt"

conversion_rates_control_df = get_df(conversion_period,(selected_experiment, 'control'))
conversion_rates_treatment_df = get_df(conversion_period,(selected_experiment, 'treatment'))

conversion_rates_df = pd.concat([conversion_rates_control_df, conversion_rates_treatment_df], ignore_index=True)
# if option == "Weekly":
#     df["time_period"] = pd.to_datetime(df["time_period"])
#df["time_period"] = pd.to_datetime(df["time_period"])
conversion_rates_fig = px.line(
    conversion_rates_df,
    x="time_period",
    y="conversion_rate",
    color="variant",          
    markers=True,

)
st.plotly_chart(conversion_rates_fig, use_container_width=True)







