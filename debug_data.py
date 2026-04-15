import pandas as pd
import os

print("Files in data folder:")
for f in os.listdir('data'):
    if f.endswith('.csv'):
        size = os.path.getsize(f'data/{f}')
        df = pd.read_csv(f'data/{f}')
        print(f"{f}: {len(df)} rows, {size} bytes")

# Check exp 1 conversion rates
exp1 = pd.read_csv('data/user_sessions.csv')
exp1_data = exp1[exp1['experiment_id'] == 1]
print(f"\nExp 1 sample: {len(exp1_data)} rows")
for var in ['control', 'treatment']:
    var_data = exp1_data[exp1_data['variant'] == var]
    users = var_data['user_id'].nunique()
    conversions = var_data.groupby('user_id')['converted'].max().sum()
    rate = conversions / users if users > 0 else 0
    print(f"  {var}: rate {rate:.4f}")

# Check the unique session_ids to see if there are old ones
all_sessions = exp1['session_id']
print(f"\nSession ID range in Exp 1: {all_sessions.min()} to {all_sessions.max()}")
exp6_sessions = exp1[exp1['experiment_id'] == 6]['session_id']
if len(exp6_sessions) > 0:
    print(f"Session ID range in Exp 6: {exp6_sessions.min()} to {exp6_sessions.max()}")
