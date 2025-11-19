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
def two_proportion_ztest(n1, x1, p1, n2, x2, p2):
    p_pool = (x1 + x2) / (n1 + n2)
    standard_error = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

