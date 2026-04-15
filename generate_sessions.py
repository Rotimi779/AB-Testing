import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read existing user sessions
sessions_df = pd.read_csv('data/user_sessions.csv')
last_session_id = sessions_df['session_id'].max()
max_user_id = sessions_df['user_id'].max()

# Read experiment assignments for experiments 6-10
assignments_df = pd.read_csv('data/experiment_assignments.csv')
exp6_10_assignments = assignments_df[assignments_df['experiment_id'].isin([6, 7, 8, 9, 10])]

new_sessions = []
session_id = last_session_id + 1

np.random.seed(42)

# Define conversion rates and characteristics for each experiment
experiment_config = {
    6: {'control': 0.145, 'treatment': 0.147, 'variance': 0.01},   # KEEP RUNNING
    7: {'control': 0.152, 'treatment': 0.128, 'variance': 0.005},  # STOP/REVERT
    8: {'control': 0.145, 'treatment': 0.150, 'variance': 0.005},  # LOW IMPACT
    9: {'control': 0.148, 'treatment': 0.155, 'variance': 0.012},  # KEEP RUNNING
    10: {'control': 0.160, 'treatment': 0.132, 'variance': 0.005}  # STOP/REVERT
}

# For each user-experiment assignment, create sessions throughout the test period
for _, assignment in exp6_10_assignments.iterrows():
    user_id = assignment['user_id']
    exp_id = assignment['experiment_id']
    variant = assignment['variant']
    assignment_date = pd.Timestamp(assignment['assignment_date'])
    
    # Generate 5-12 sessions per user across the test period
    num_sessions = np.random.randint(5, 13)
    
    # Get conversion rate for this variant
    config = experiment_config[exp_id]
    base_rate = config[variant]
    variance = config['variance']
    
    for _ in range(num_sessions):
        # Random date between assignment and end of test
        days_offset = np.random.randint(0, 90)
        session_date = assignment_date + timedelta(days=days_offset)
        
        # Session characteristics
        page_views = np.random.randint(1, 15)
        time_on_site = np.random.randint(30, 600)
        clicked_cta = np.random.choice([True, False], p=[0.65, 0.35])
        
        # Conversion with some variance
        actual_rate = base_rate + np.random.normal(0, variance)
        actual_rate = max(0, min(1, actual_rate))  # Clamp between 0 and 1
        converted = np.random.random() < actual_rate
        
        # Revenue only if converted
        revenue = 0.0
        if converted:
            revenue = round(np.random.uniform(15, 150), 2)
        
        new_sessions.append({
            'session_id': session_id,
            'user_id': user_id,
            'experiment_id': exp_id,
            'session_date': session_date.strftime('%Y-%m-%d'),
            'variant': variant,
            'page_views': page_views,
            'time_on_site': time_on_site,
            'clicked_cta': clicked_cta,
            'converted': converted,
            'revenue': revenue
        })
        session_id += 1

new_df = pd.DataFrame(new_sessions)
print(f'Created {len(new_df)} new session rows')
print(f'Sample rows:')
print(new_df.head(10))

# Append to existing file
combined_df = pd.concat([sessions_df, new_df], ignore_index=True)
combined_df.to_csv('data/user_sessions.csv', index=False)
print(f'\nTotal user_sessions rows now: {len(combined_df)}')
print(f'Conversion rates by experiment:')
for exp_id in [6, 7, 8, 9, 10]:
    exp_data = new_df[new_df['experiment_id'] == exp_id]
    for variant in ['control', 'treatment']:
        var_data = exp_data[exp_data['variant'] == variant]
        if len(var_data) > 0:
            conv_rate = var_data['converted'].mean()
            print(f'  Exp {exp_id} {variant}: {conv_rate:.4f}')
