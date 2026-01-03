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

# col1,col2 = st.columns(2)
# with col1:
#     experiment = st.selectbox('Select an experiment',['Checkout Button Color', 'Pricing Display Test', 'Email Subject Line', 'Product Page Layout', 'Free Shipping Threshold'])

experiment = st.selectbox('Select an experiment',['Checkout Button Color', 'Pricing Display Test', 'Email Subject Line', 'Product Page Layout', 'Free Shipping Threshold'])
selected_experiment = experiment_mapping_dict[experiment]


# with col2:
#     time_option = st.selectbox(
#         "Choose a time period",
#         ["Weekly", "Monthly"]
#     )

experiments_results_summary = 'experiments_results_summary.txt'
experiments_results_summary_df = get_df(experiments_results_summary,(selected_experiment,))
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Lift", f"{experiments_results_summary_df['lift_percent'].iloc[0]:+.2f}%" if experiments_results_summary_df["lift_percent"] is not None else "N/A")
metric2.metric("p-value", f"{experiments_results_summary_df['p_value'].iloc[0]:.4g}")
metric3.metric("z-score", f"{experiments_results_summary_df['z_score'].iloc[0]:.3f}")
metric4.metric("Significant?", "Yes ✅" if experiments_results_summary_df["p_value"].iloc[0] < 0.05 else "No ❌")



##CONFIDENCE INTERVAL PLOTS SHOULD BE RIGHT HERE!!!!!!!!
st.subheader("Confidence Interval")
difference = experiments_results_summary_df['treatment_rate'].iloc[0] - experiments_results_summary_df['control_rate'].iloc[0]
fig = ci_plot(difference, experiments_results_summary_df["lower_ci"].iloc[0], experiments_results_summary_df["upper_ci"].iloc[0],experiments_results_summary_df["p_value"].iloc[0])
st.plotly_chart(fig, use_container_width=True)

#Start with thoughts on values like lift,p-value,z-score and how they tie into significance for each of the readings
#Also at the end, put suggestions for how to continue with each experiments based on the results, like if you should continue for more sample data or scrap it
# or end it due to it being inconsequential or having more than enough data
control_rate = experiments_results_summary_df['control_rate'].iloc[0] * 100
treatment_rate = experiments_results_summary_df['treatment_rate'].iloc[0] * 100
lift = experiments_results_summary_df['lift_percent'].iloc[0]

metric_types = {'conversion': 'conversion', 'revenue': 'revenue', 'click_through': 'click through', 'engagement': 'engagement'}

metric_to_track = metric_types[experiments_results_summary_df['metric_type'].iloc[0]]

p_value = experiments_results_summary_df["p_value"].iloc[0]
#Lower limit and upper limit for these experiments to see just how close their p-values were to 0.05
lower_limit_to_experiment = 0.05 * 0.8
upper_limit_to_experiment = 0.05 * 1.2
#Also talk about how the confidnce interval affects significance, using the rule that if lower_ci < 0 < upper_ci then not significant,
#else if lower_ci > 0 or upper_ci < 0 then significant.
if p_value >= lower_limit_to_experiment and p_value <= 0.05:
    p_value_insight = """The p-value of this experiment is slightly less than 0.05. While the experiment is \
    statistically significant, the p-value is still very close to 0.05(slightly below).
    """
    conclusion = """ Ler
    """
elif p_value <= upper_limit_to_experiment and p_value > 0.05:
    p_value_insight = """The p-value of this experiment is slightly greater than 0.05. While the experiment is not statistically \
    significant, the p-value is still very close to 0.05(slightly above).
    """
elif p_value < 0.05:
    p_value_insight = """The p-value of this experiment is less than 0.05. Therefore, the experiment is statistically significant and it is unlikely that the change in performance \
    is due to chance. The change from the control subgroup to treatment subgroup definitely had an effect on performance.
    """
else:
    p_value_insight = """The p-value of this experiment is greater than 0.05. Therefore, the experiment is not statistically significant and it is likely that the change in performance \
    is due to chance. The change from the control subgroup to treatment subgroup does not have enough evidence to claim that there was an effect on performance.
    """

lower_ci = experiments_results_summary_df['lower_ci'].iloc[0]
upper_ci = experiments_results_summary_df['upper_ci'].iloc[0]

if (lower_ci < 0 and upper_ci > 0) or lower_ci == 0 or upper_ci == 0:
    ci_insight = f"""The confidence interval of this experiment ranges from {lower_ci} to {upper_ci}. Due to the boundary from the
    lower level to the upper level crossing 0, the experiment is proven to not be statistically significant, further supporting the 
    claim due to the p-value.The change from the control group to the treatment group does not have enough evidence to claim that there was an effect on performance of the
    {metric_to_track}.
    """
elif lower_ci > 0 or upper_ci < 0:
    ci_insight = f"""The confidence interval of this experiment ranges from {lower_ci} to {upper_ci}. The boundaries of this experiment 
    do not cross 0, therefore this experiment is statistically significant. The change from the control group to the treatment group
    had an effect on performance of the {metric_to_track}.
    """

z_score = round(experiments_results_summary_df['z_score'].iloc[0],4)

if z_score >= 1.96:
    z_score_insight = f"""The z-score for this experiment is {z_score}. Due to being greater than 1.96, this proves the 
    decision that the change from the control group to the treatment group had an effect on performance of the {metric_to_track}.
    """
else:
    z_score_insight = f"""The z-score for this experiment is {z_score}. Due to being less than 1.96, this proves the 
    decision that the change from the control group to the treatment group does not have enough evidence 
    to claim there was an effect on performance of the {metric_to_track}.
    """




st.info(
    f"""
        **Experiment**: {experiment}\
        \n**Hypothesis**: {experiments_results_summary_df['hypothesis'].iloc[0]}, testing for {metric_to_track}.\
        \n\nFor this experiment, the initial conversion rate for the controlled scenario was **{control_rate}**% and the conversion rate for the treatment
        scenario was **{treatment_rate}**%. This experiment experienced a lift of {lift}%.\
        \n{p_value_insight}\
        \n{ci_insight}\
        \n{z_score_insight}\
        \n***Final verdict***: 
    """
)




# with col2:
    # time_option = st.selectbox(
    #     "Choose a time period",
    #     ["Weekly", "Monthly"]
    # )


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







