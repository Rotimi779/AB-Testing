import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import zt_ind_solve_power
from math import *

user_sessions_df = pd.read_csv('data/user_sessions.csv')


#Phase 1: Exploring data and running some hypothesis tests

#Just exploring!!!!
# print(user_sessions_df.head(10))

# print(user_sessions_df.describe())
# print(("\n"))

#How many total sessions do we have?
print('Total sessions:', user_sessions_df['session_id'].nunique())

#Hmmm, what about the experiments? Let's check how many experiments we have
print('Total experiments:', user_sessions_df['experiment_id'].nunique())

#Now let's check the conversion rates for these sessions, starting with the average conversion rate then the conversion rates per person
#Take a look at this side next time
overall_conversion =  user_sessions_df.groupby('user_id')
average_conversion = (overall_conversion['converted'].max() / len(overall_conversion)).mean()
#print('Average conversion rate:', average_conversion)
# #Pretty low average conversion rate, but let's see if any of the experiments did better

# conversion = 
# conversion_rates_by_experiment = user_sessions_df.groupby('experiment_id')['converted'].mean()
# print('Conversion rates by experiment:', conversion_rates_by_experiment)
# print("\n\n")
#They are about the same, just about 2% conversion rate for all experiments. Not too good.

#Lets take a look at the conversion rates of the control group vs the treatment groups of experiment 1:button_color
button_color_control = user_sessions_df[(user_sessions_df['experiment_id'] == 1) & (user_sessions_df['variant'] == 'control')]
button_color_treatment = user_sessions_df[(user_sessions_df['experiment_id'] == 1) & (user_sessions_df['variant'] == 'treatment')]
button_color_control_conversion = button_color_control.groupby('user_id')['converted'].max().sum()
button_color_treatment_conversion = button_color_treatment.groupby('user_id')['converted'].max().sum()
button_color_control_sample_size = len(button_color_control.groupby('user_id')['converted'].max())
button_color_treatment_sample_size = len(button_color_treatment.groupby('user_id')['converted'].max())
button_color_control_rate = button_color_control_conversion / button_color_control_sample_size
button_color_treatment_rate = button_color_treatment_conversion / button_color_treatment_sample_size
print("\n\n")
print(button_color_control_rate)
print("Experiment 1 Control group Conversion Rates:", button_color_control_rate.mean())
print("Experiment 1 Treatment group Conversion Rates:", button_color_treatment_rate.mean())
#print(f"The treatment group has a higher conversion rate, with an absolute difference of {round(button_color_treatment_rate - button_color_control_rate,4)} ")
print(f"The percentage improvement(relative lift) between these two is {round((button_color_treatment_rate - button_color_control_rate)/button_color_control_rate * 100,2)}%")
print("\n\n")
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



button_color_results = two_proportion_ztest(button_color_control_sample_size,button_color_control_conversion, button_color_treatment_sample_size, button_color_treatment_conversion)
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

    #Aggregate to user level
    control_df_conversions = control_df.groupby('user_id')['converted'].max()
    treatment_df_conversions = treatment_df.groupby('user_id')['converted'].max()

    #Count users and conversions
    control_n = len(control_df_conversions)
    treatment_n = len(treatment_df_conversions)
    control_x = control_df_conversions.sum()
    treatment_x = treatment_df_conversions.sum()
    
    # print(f"Control_n is {control_n}. Control_x is {control_x}. Treatment_n is {treatment_n} and treatment_x is {treatment_x}")
    # break
    control_rate = round(control_x/control_n, 2)
    treatment_rate = round(treatment_x/treatment_n, 2)
    results = two_proportion_ztest(control_n,control_x,treatment_n,treatment_x)
    print(f"Experiment {index}: {item}")
    print(f"Control Conversion Rate: {control_rate} ({control_x}/{control_n})")
    print(f"Treatment Conversion Rate: {treatment_rate} ({treatment_x}/{treatment_n})")
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
#free_shipping_threshold has a low p-value(REJECT H0) and is significant. Very close to not being significant. 9.58% lift also makes it economically beneficial. Lower the shipping threshold, but discuss with stakeholders and maybe launch to a percentagfe of users(maybe 30%), and monitor progress.


#Phase 2: Time to look at sample sizes and power analysis. We want to know the ideal number of users needed for our tests

#Key definitions: 
#1.) Sample users: Number of users in each group. More users means more sensitive tests.
#2.) Effect size: The minimum detectable effect. Would be the smallest difference you want to be able to detect like  a 3% lift from controlled experiment to treatment experiment
#3.) Statistical power: The probability of detecting a real effect when it exists, like the ability to detect a 5% lift.
#4.) Significance level: The probability of incorrectly rejecting a true null hypothesis in a statistical test(chance of a false positive)

#Let's start with getting the power for each of our experiments
def calculate_statistical_power(summary_df,user_sessions_df):
    control_count = user_sessions_df[user_sessions_df['variant'] == 'control']['user_id'].nunique()
    treatment_count = user_sessions_df[user_sessions_df['variant'] == 'treatment']['user_id'].nunique()
    required_nobs1 = zt_ind_solve_power(effect_size=summary_df['lift_percent'], alpha=0.05, power =None,nobs=control_count, ratio=1, alternative='two-sided')
    return required_nobs1

#WORKING HERE AS OF 7:44 pm on 23rd Nov. 2025. COME STRAIGHT HERE AND FINISH THIS
print("\n\nBANKAI\n")
print(calculate_statistical_power(results_summary_df[results_summary_df['experiment_name'] == 'pricing_display_test'],user_sessions_df[]))