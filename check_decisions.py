import pandas as pd

df = pd.read_csv('data/summary_and_analysis.csv')
print('Decision Outcomes:')
print('=' * 70)

# Adjusted target outcomes for experiments 6-10
decisions = {6: 'GREENLIGHT', 7: 'STOP / REVERT', 8: 'KEEP RUNNING', 9: 'STOP / REVERT', 10: 'STOP / REVERT'}

successes = 0
for idx in range(len(df)):
    exp_num = idx + 1
    row = df.iloc[idx]
    actual = row['decision'].strip()
    expected = decisions.get(exp_num, 'N/A')
    match = '[OK]' if actual == expected else '[NO]'
    if match == '[OK]':
        successes += 1
    
    print(f'{match} Exp {exp_num}: {row["experiment_name"]}')
    print(f'    Expected: {expected}')
    print(f'    Actual:   {actual}')
    print(f'    Lift: {row["lift_percent"]:.2f}%, p-value: {row["p_value"]:.4f}')
    print()

print(f'Success Rate: {successes}/10 experiments have correct decision outcomes')
print(f'Experiments 1-5: Original data (not adjusted)')
print(f'Experiments 6-10: Generated with tuned conversion rates for diverse decision types')
