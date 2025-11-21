import pandas as pd
import numpy as np
from scipy import stats
from math import *

user_sessions_df = pd.read_csv('data/user_sessions.csv')

#Just exploring!!!!
# print(user_sessions_df.head(10))

# print(user_sessions_df.describe())
# print(("\n"))

#How many total sessions do we have?
print('Total sessions:', user_sessions_df['session_id'].nunique())

#Hmmm, what about the experiments? Let's check how many experiments we have
print('Total experiments:', user_sessions_df['experiment_id'].nunique())

#Now let's check the conversion rates for these sessions, starting with the overall conversion rate then the conversion rates per experiment
overall_conversion_rate =  user_sessions_df['converted']
print('Overall conversion rate:', overall_conversion_rate.mean())
#Pretty low conversion rate overall, but let's see if any of the experiments did better
conversion_rates_by_experiment = user_sessions_df.groupby('experiment_id')['converted'].mean()
print('Conversion rates by experiment:', conversion_rates_by_experiment)
print("\n\n")
#They are about the same, just about 2% conversion rate for all experiments. Not too good.

#Lets take a look at the conversion rates of the control group vs the treatment groups of experiment 1:button_color
button_color_control = user_sessions_df[(user_sessions_df['experiment_id'] == 1) & (user_sessions_df['variant'] == 'control')]
button_color_treatment = user_sessions_df[(user_sessions_df['experiment_id'] == 1) & (user_sessions_df['variant'] == 'treatment')]
button_color_control_mean = button_color_control['converted'].mean()
button_color_treatment_mean = button_color_treatment['converted'].mean()
print("Experiment 1 Control group Conversion Rates:", button_color_control_mean)
print("Experiment 1 Treatment group Conversion Rates:", button_color_treatment_mean)
print(f"The treatment group has a higher conversion rate, with an absolute difference of {round(button_color_treatment_mean - button_color_control_mean,4)} ")
print(f"The percentage improvement(relative lift) between these two is {round((button_color_treatment_mean - button_color_control_mean)/button_color_control_mean * 100,2)}%")

#The treatment group seems to have a higher conversion rate in general


#Now time for the real work. Let's make a function for running the two-proportion z-test on our data
def two_proportion_ztest(n1, x1, n2, x2):
    """
    Perform two-proportion z-test.
    
    Parameters:
    n1: sample size for group 1 (control)
    x1: number of successes in group 1 (control conversions)
    n2: sample size for group 2 (treatment)
    x2: number of successes in group 2 (treatment conversions)
    
    Returns:
    dict with z_score, p_value, significance (boolean), lower confidence interval, upper confidence interval and lift_percent
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    standard_error = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z_score = (p2 - p1) / standard_error
    p_value_two_tail = 2 * stats.norm.sf(abs(z_score))

    difference = p2 - p1
    standard_error_diff = sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
    confidence_interval_lower = difference - 1.96 * standard_error_diff
    confidence_interval_upper = difference + 1.96 * standard_error_diff
    lift_percent = round((p2-p1)/p1 * 100,2)
    return {'z_score':z_score, 'p_value':p_value_two_tail, 'significant': p_value_two_tail < 0.05, 'lower_ci': confidence_interval_lower, 'upper_ci': confidence_interval_upper, 'lift_percent':lift_percent }

c_length = len(button_color_control)
c_conversion_length = len(button_color_control[button_color_control['converted'] == True])
t_length = len(button_color_treatment)
t_conversion_length = len(button_color_treatment[button_color_treatment['converted'] == True])
button_color_results = two_proportion_ztest(c_length,c_conversion_length, t_length, t_conversion_length)
print(f"\n\nThe button color results are {button_color_results}\n")

experiments_df = list(pd.read_csv('data/experiments.csv')['experiment_name'])
experiments = {}
for i in range(1,6):
    experiments[i] = experiments_df[i - 1]
print(f'{experiments}\n')
results_summary_df = pd.DataFrame()

for index,item in experiments.items():
    df = user_sessions_df[user_sessions_df['experiment_id'] == index]
    control_df = df[df['variant'] == 'control']
    treatment_df = df[df['variant'] == 'treatment']
    control_df_length = len(control_df)
    treatment_df_length = len(treatment_df)
    control_conversion_length = len(control_df[control_df['converted'] == True])
    treatment_conversion_length = len(treatment_df[treatment_df['converted'] == True])
    control_rate = round(control_conversion_length/control_df_length * 100,2)
    treatment_rate = round(treatment_conversion_length/treatment_df_length * 100,2)
    results = two_proportion_ztest(control_df_length,control_conversion_length,treatment_df_length,treatment_conversion_length)
    print(f"Experiment {index}: {item}")
    print(f"Control Conversion Rate: {round(control_conversion_length/control_df_length * 100,2)} ({control_conversion_length}/{control_df_length})")
    print(f"Treatment Conversion Rate: {round(treatment_conversion_length/treatment_df_length * 100,2)} ({treatment_conversion_length}/{treatment_df_length})")
    print(f"Lift_percent: {results['lift_percent']}")
    print(f"Z-score: {results['z_score']}")
    print(f"P-value: {results['p_value']}")
    if results['p_value'] < 0.05:
        print('Decision: REJECT H0 - Result is STASTISTICALLY SIGNIFICANT')
    else:
        print('Decision: FAIL TO REJECT H0 - Not enough evidence')
    print(f"95% Confidence Interval Limits: [{round(results['lower_ci'],6)}%, {round(results['upper_ci'],6)}%]")
    print("\n")
    new_row = pd.DataFrame({'experiment_name':[item], 'control_rate':[control_rate], 'treatment_rate':[treatment_rate],'lift_percent':[results['lift_percent']],'z_score':[results['z_score']],'p_value':[results['p_value']],'is_significant': [results['p_value'] < 0.05]})
    results_summary_df = pd.concat([results_summary_df,new_row])


print(results_summary_df)
results_summary_df.to_csv('data/results_summary.csv')
#With these results, checkout_button_color has a very low p-value(REJECT H0) and is highly significant. Business recommendation would be to use the new button color immediately
#pricing_display_test has a very high p-value(FAIL TO REJECT H0) and is not close to significant. Business recommendation would be not to display discount percentages
#email_subject_line has a  high p-value(FAIL TO REJECT H0) and is not close to significant. Business recommendation would be not to implement personalized subject lines.
#product_page_layout has a high enough p-value(FAIL TO REJECT H0) and is not significant, although very close to that. In this case, lift_percent is negative, meaning that it's economically better to use a list layout over grid layout; 7% could be crucial
#free_shipping_threshold has a low p-value(REJECT H0) and is significant. Very close to not being significant. 8.54% lift also makes it economically beneficial. Lower the shipping threshold, but discuss with stakeholders and maybe launch to a percentagfe of users(maybe 30%), and monitor progress.