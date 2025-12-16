from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

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

st.set_page_config(page_title="A/B Testing Analysis", layout="wide")
st.title("A/B Testing Dashboard")

st.header("Experiment Analysis")
st.info(
    """
    **Experiment Overview**

    This section summarizes the performance of ongoing A/B experiments by comparing
    control and treatment variants across key metrics. For each experiment, we analyze
    **user-based conversion rates over time**, visualize **confidence intervals** around
    observed lifts, and report **statistical test results** including lift percentage,
    z-scores, and p-values.

    These views together help assess both the **magnitude** and **statistical reliability**
    of observed differences, enabling informed decisions on whether an experiment shows
    meaningful impact.
    """
)

st.subheader("Conversion Rates Over Time")

experiment_mapping_dict = {'Checkout Button Color':1, 'Pricing Display Test':2, 'Email Subject Line': 3, 'Product Page Layout': 4, 'Free Shipping Threshold': 5}

col1,col2 = st.columns(2)
with col1:
    experiment = st.selectbox('Select an experiment',['Checkout Button Color', 'Pricing Display Test', 'Email Subject Line', 'Product Page Layout', 'Free Shipping Threshold'])
selected_experiment = experiment_mapping_dict[experiment]


with col2:
    time_option = st.selectbox(
        "Choose a time period",
        ["Weekly", "Monthly"]
    )

experiments_results_summary = 'experiments_results_summary.txt'
experiments_results_summary_df = get_df(experiments_results_summary,(selected_experiment,))
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Lift", f"{experiments_results_summary_df['lift_percent'].iloc[0]:+.2f}%" if experiments_results_summary_df["lift_percent"] is not None else "N/A")
metric2.metric("p-value", f"{experiments_results_summary_df['p_value'].iloc[0]:.4g}")
metric3.metric("z-score", f"{experiments_results_summary_df['z_score'].iloc[0]:.3f}")
metric4.metric("Significant?", "Yes ✅" if experiments_results_summary_df["p_value"].iloc[0] < 0.05 else "No ❌")



##CONFIDENCE INTERVAL PLOTS SHOULD BE RIGHT HERE!!!!!!!!









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
    markers=True
)
st.plotly_chart(conversion_rates_fig, use_container_width=True)







