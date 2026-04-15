import pandas as pd

df = pd.read_csv('data/user_sessions.csv')
print(f"Total rows in user_sessions: {len(df)}")
print(f"Experiments in database: {sorted(df['experiment_id'].unique())}")

for exp_id in range(6, 11):
    exp = df[df['experiment_id'] == exp_id]
    print(f"\nExp {exp_id}: {len(exp)} rows")
    for var in ['control', 'treatment']:
        var_data = exp[exp['variant'] == var]
        users = var_data['user_id'].nunique()
        conversions = var_data.groupby('user_id')['converted'].max().sum()
        rate = conversions / users if users > 0 else 0
        print(f'  {var}: {users} users, {conversions} conversions, rate {rate:.4f}')
