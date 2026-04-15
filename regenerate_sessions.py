import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read existing user sessions
sessions_df = pd.read_csv('data/user_sessions.csv')
# Remove the experiments 6-10 data that was just added (to regenerate with correct rates)
sessions_df = sessions_df[sessions_df['experiment_id'] <= 5]

last_session_id = sessions_df['session_id'].max()
max_user_id = sessions_df['user_id'].max()

# Read experiment assignments for experiments 6-10
assignments_df = pd.read_csv('data/experiment_assignments.csv')
exp6_10_assignments = assignments_df[assignments_df['experiment_id'].isin([6, 7, 8, 9, 10])]

new_sessions = []
session_id = last_session_id + 1

np.random.seed(42)

# Corrected conversion rates (should be in 14-15% range like original experiments)
# Exp 6 (search_result_sorting): LOW IMPACT - tiny positive effect, p < 0.05
#   Need conversion rates like: control 14.5%, treatment 14.65% (0.15pp positive)
# Exp 7 (checkout_form_fields): STOP/REVERT - significant negative
#   Need: control 14.8%, treatment 12.5% (2.3pp negative to get p < 0.05)
# Exp 8 (recommendation_algorithm): KEEP RUNNING - no significant difference
#   Need: control 14.3%, treatment 14.2% (small noise)
# Exp 9 (cart_discount_display): LOW IMPACT - tiny positive effect, p < 0.05
#   Need: control 14.6%, treatment 14.75% (0.15pp positive)
# Exp 10 (mobile_app_onboarding): STOP/REVERT - significant negative (should be good already)
#   Need: control 15.0%, treatment 12.0% (3.0pp negative)
experiment_config = {
    # Exp 6: GREENLIGHT - 2.3pp positive effect (stronger for p < 0.05)
    #   control 14.5%, treatment 16.8% (2.3pp positive effect)
    6: {'control': 0.1450, 'treatment': 0.1680, 'variance': 0.0002},
    
    # Exp 7: STOP/REVERT - clear negative effect, p < 0.05
    #   control 14.8%, treatment 12.0% (2.8pp negative effect)
    7: {'control': 0.1480, 'treatment': 0.1200, 'variance': 0.0030},
    
    # Exp 8: KEEP RUNNING - no real effect, high variance
    #   control 14.3%, treatment 14.3% (0% effect, high noise)
    8: {'control': 0.1430, 'treatment': 0.1430, 'variance': 0.0100},
    
    # Exp 9: STOP/REVERT - negative effect, p < 0.05
    #   control 14.8%, treatment 11.8% (3.0pp negative effect)
    9: {'control': 0.1480, 'treatment': 0.1180, 'variance': 0.0030},
    
    # Exp 10: STOP/REVERT - strong negative effect, p < 0.05
    #   control 15.0%, treatment 12.0% (3.0pp negative effect)
    10: {'control': 0.1500, 'treatment': 0.1200, 'variance': 0.0030}
}

# For each user-experiment assignment, create sessions throughout the test period
for _, assignment in exp6_10_assignments.iterrows():
    user_id = assignment['user_id']
    exp_id = assignment['experiment_id']
    variant = assignment['variant']
    assignment_date = pd.Timestamp(assignment['assignment_date'])
    
    # Generate 5-12 sessions per user
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
        
        # Conversion with variance
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
print(f'\nConversion rates by experiment (should match targets):')
for exp_id in [6, 7, 8, 9, 10]:
    exp_data = new_df[new_df['experiment_id'] == exp_id]
    for variant in ['control', 'treatment']:
        var_data = exp_data[exp_data['variant'] == variant]
        if len(var_data) > 0:
            conv_rate = var_data['converted'].mean()
            print(f'  Exp {exp_id} {variant}: {conv_rate:.4f} ({conv_rate*100:.2f}%)')

# Append to original data
combined_df = pd.concat([sessions_df, new_df], ignore_index=True)
combined_df.to_csv('data/user_sessions.csv', index=False)
print(f'\nTotal user_sessions rows now: {len(combined_df)}')
