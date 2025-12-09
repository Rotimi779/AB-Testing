import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import fsolve
from statsmodels.stats.power import zt_ind_solve_power
from math import *
from statsmodels.stats.proportion import proportion_effectsize

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
    control_rate = round(control_x/control_n, 4)
    treatment_rate = round(treatment_x/treatment_n, 4)
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
    new_row = pd.DataFrame({'experiment_name':[item], 'control_rate':[control_rate],'control_size':[control_n], 'treatment_rate':[treatment_rate], 'treatment_size':treatment_n,'lift_percent':[results['lift_percent']],'z_score':[results['z_score']],'p_value':[results['p_value']],'is_significant': [results['p_value'] < 0.05]})
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
def calculate_statistical_power_from_results(summary_df):
    p1 = summary_df['control_rate']
    p2 = summary_df['treatment_rate']
    effect_size = proportion_effectsize(p2, p1).iloc[0]
    print(f"The effect soze  is {effect_size}")
    nobs1 = summary_df['control_size']
    control_n = summary_df['control_size']
    treatment_n = summary_df['treatment_size']
    ratio = treatment_n.iloc[0] / control_n.iloc[0]
    control_count = user_sessions_df[user_sessions_df['variant'] == 'control']['user_id'].nunique()
    treatment_count = user_sessions_df[user_sessions_df['variant'] == 'treatment']['user_id'].nunique()
    #Maybe change the ratio here too
    power = zt_ind_solve_power(effect_size=effect_size, alpha=0.05 ,nobs1=nobs1, ratio=ratio, alternative='two-sided')
    return power

print("\n\nBANKAI\n")
#TAKE A VERY GOOD LOOK AT HOW YOURE DOING THE two proportion z test function. It MAY BE OFF. CHECKED AS OF 12:59 am 26/11/25
for index,items in experiments.items():
    power = calculate_statistical_power_from_results(results_summary_df[results_summary_df['experiment_name'] == items])
    print(f"The power for the experiment {items} is {power}\n")



#Now let's try and get the number of users we would need to detect a lift(sample size)
def calculate_sample_users_from_results(summary_df):
    """
    Calculates the number of users per group required to detect a lift, the total number of users based on the ratio and the multiplier to reach
    the required users per group
    """
    p1 = summary_df['control_rate']
    p2 = summary_df['treatment_rate']
    control_n = summary_df['control_size'].iloc[0]
    treatment_n = summary_df['treatment_size'].iloc[0]
    effect_size = proportion_effectsize(p2, p1).iloc[0]
    ratio = treatment_n / control_n
    
    required_users = int(round(zt_ind_solve_power(effect_size=effect_size,alpha=0.05,power=0.8,ratio=ratio, alternative='two-sided'),0))

    total_current = control_n + treatment_n
    total_required = int(round(required_users * (1 + ratio),0))
    multiplier = round(required_users / control_n,1)

    #Problem with these print statements here!!
    # print(f"\n{summary_df['experiment_name'].iloc[0]}:")
    # print(f"The effect size  is {effect_size}")
    # print(f"Current: Control={control_n:,}, Treatment={treatment_n:,} (Total: {total_current:,})")
    # print(f"Required: {required_users:,.0f} per control group")
    # print(f"Total required: {total_required:,.0f} users")
    # print(f"Need {multiplier:.1f}x more users")
    return required_users, total_required, multiplier
#Will add this to dataframe which will be formatted to csv 
for index,items in experiments.items():
    print(calculate_sample_users_from_results(results_summary_df[results_summary_df['experiment_name'] == items]))



#Now it's time to work on an effect size calculator for a minimum detectable effect


def calculate_minimum_detectable_effect_from_results(summary_df):
    """
    Calculates the minimum lift you could have detected from control conversion to treatment conversion 
    """
    p1 = summary_df['control_rate']
    p2 = summary_df['treatment_rate']
    control_n = summary_df['control_size'].iloc[0]
    treatment_n = summary_df['treatment_size'].iloc[0]
    ratio = treatment_n / control_n
    nobs1 = summary_df['control_size']

    effect_size = zt_ind_solve_power(effect_size=None,nobs1=nobs1,alpha=0.05,power=0.8,ratio=ratio, alternative='two-sided')
    
    def equation(p2):
        return (2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(control_rate))) - effect_size

    mde_treatment_rate = fsolve(equation, control_rate * 1.1)[0]
    mde_relative_lift = (mde_treatment_rate - control_rate) / control_rate * 100

    return {
        'effect_size': effect_size,
        'baseline_rate': p1, #Control rate
        'mde_treatment_rate': mde_treatment_rate,  # THEORETICAL minimum
        'mde_relative_lift_pct': mde_relative_lift,  # MDE as %
        'actual_treatment_rate': summary_df['treatment_rate'].iloc[0],  # ACTUAL from data
        'actual_lift_pct': summary_df['lift_percent'].iloc[0],  # ACTUAL lift
        'sample_size': control_n
    }

print("\nNow time for MDE\n")
#Problem with mde displayed as all of them are just on 12% foe mde we would think we could detect. FIX IT!!!    
for index,items in experiments.items():
    print(f"This is for the experiment {items}")
    mde = calculate_minimum_detectable_effect_from_results(results_summary_df[results_summary_df['experiment_name'] == items])
    print(f"   MDE (what you COULD detect): {mde['mde_relative_lift_pct']:.1f}%")
    print(f"   Actual (what you DID observe): {mde['actual_lift_pct']:.1f}%")
    print(f"   ")
    if abs(mde['actual_lift_pct']) >= abs(mde['mde_relative_lift_pct']):
        print(f"   ✅ Actual lift > MDE → Well-powered")
    else:
        print(f"   ❌ Actual lift < MDE → Underpowered")
    print("\nOnto the next\n")
    
    # print(calculate_minimum_detectable_effect_from_results(results_summary_df[results_summary_df['experiment_name'] == items]))





#Charts to show
#Dropdown menu for each experiment(can also do control vs treatment)
#   Start with conversion rates over time.
#   Confidence interval visualization
#   Lift values, p values, z scores
#   Analysis based on these values
#   Provide suggesttions for businesses to move forward and what experiments to scrap. Export results as pdf

#Calculator for detecting sample size, power and effect size 
# An option for using the some of the values that we have in the csv files and maybe one where you can fill in all the values yourself(user will input
# what they want to check for)
# A section for actual analysis, where we state what we think went wrong with the experiment. Provide suggesttions for businesses 
# to move forward and what experiments to scrap. Could include how long(months or years) it will take to reach a certain lift or how many more people we would need
# to see a good change

#A section where we use the csv readings and give out analysis based on 80% power and 0.05 significance level. Make sure to state this
#This section would not change no matter what. Provide suggesttions for businesses to move forward and what experiments to scrap

#Last chart would be on segmentation analysis(or could be before calculator)

print("\nWE GOOD WITH ALL OF IT\n")
