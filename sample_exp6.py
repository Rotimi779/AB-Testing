import pandas as pd

# Check actual converted values for exp 6
df = pd.read_csv('data/user_sessions.csv')
exp6 = df[df['experiment_id'] == 6]

# Sample a few rows
print("Sample rows from Exp 6:")
print(exp6[['session_id', 'user_id', 'converted']].head(30))

# Check actual conversion counts
for var in ['control', 'treatment']:
    var_data = exp6[exp6['variant'] == var]
    num_rows = len(var_data)
    num_converted = var_data['converted'].sum()
    print(f"\n{var}: {num_converted} session conversions out of {num_rows} sessions ({num_converted/num_rows:.2%})")
    
    # Check user-level
    user_conv = var_data.groupby('user_id')['converted'].max().sum()
    users = var_data['user_id'].nunique()
    print(f"       {user_conv} user conversions out of {users} users ({user_conv/users:.2%})")
