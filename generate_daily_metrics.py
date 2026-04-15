import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read existing daily metrics
daily_df = pd.read_csv('data/daily_metrics.csv')

# Generate daily metrics for experiments 6-10
new_daily_metrics = []
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 3, 31)

np.random.seed(42)

# Experiment 6 (search_result_sorting): KEEP RUNNING
# Control ~14.5%, Treatment ~14.7% (very small diff, inconclusive)
exp6_control_rate = 0.1450
exp6_treatment_rate = 0.1470
exp6_variance = 0.01

# Experiment 7 (checkout_form_fields): STOP/REVERT
# Control ~15.2%, Treatment ~12.8% (significant degradation)
exp7_control_rate = 0.1520
exp7_treatment_rate = 0.1280

# Experiment 8 (recommendation_algorithm): LOW IMPACT
# Control ~14.5%, Treatment ~15.0% (small but significant)
exp8_control_rate = 0.1450
exp8_treatment_rate = 0.1500

# Experiment 9 (cart_discount_display): KEEP RUNNING
# Control ~14.8%, Treatment ~15.5% (moderate, but noisy)
exp9_control_rate = 0.1480
exp9_treatment_rate = 0.1550
exp9_variance = 0.012

# Experiment 10 (mobile_app_onboarding): STOP/REVERT
# Control ~16.0%, Treatment ~13.2% (significant degradation)
exp10_control_rate = 0.1600
exp10_treatment_rate = 0.1320

current_date = start_date
while current_date <= end_date:
    # For each experiment and variant
    for exp_id, rates in [
        (6, (exp6_control_rate, exp6_treatment_rate, exp6_variance)),
        (7, (exp7_control_rate, exp7_treatment_rate, 0.005)),
        (8, (exp8_control_rate, exp8_treatment_rate, 0.005)),
        (9, (exp9_control_rate, exp9_treatment_rate, exp9_variance)),
        (10, (exp10_control_rate, exp10_treatment_rate, 0.005))
    ]:
        control_rate, treatment_rate, variance = rates
        
        # Add noise to rates
        control_rate += np.random.normal(0, variance)
        treatment_rate += np.random.normal(0, variance)
        
        for variant, rate in [('control', control_rate), ('treatment', treatment_rate)]:
            # Generate daily users (600-700 per variant per day)
            total_users = np.random.randint(600, 700)
            total_conversions = int(total_users * max(0, rate))
            actual_rate = total_conversions / total_users if total_users > 0 else 0
            
            # Revenue: 1.5-4.0 per user
            total_revenue = total_users * np.random.uniform(1.5, 4.0)
            avg_revenue = total_revenue / total_users
            
            new_daily_metrics.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'experiment_id': exp_id,
                'variant': variant,
                'total_users': total_users,
                'total_conversions': total_conversions,
                'conversion_rate': round(actual_rate, 4),
                'total_revenue': round(total_revenue, 2),
                'avg_revenue_per_user': round(avg_revenue, 2)
            })
    
    current_date += timedelta(days=1)

new_df = pd.DataFrame(new_daily_metrics)
print(f'Created {len(new_df)} new daily metric rows')
print(f'Date range: {new_df["date"].min()} to {new_df["date"].max()}')

# Append to existing file
combined_df = pd.concat([daily_df, new_df], ignore_index=True)
combined_df.to_csv('data/daily_metrics.csv', index=False)
print(f'Total daily_metrics rows now: {len(combined_df)}')
