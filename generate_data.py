"""
A/B Testing Platform - Synthetic Data Generator
Generates realistic e-commerce A/B testing data with proper statistical properties
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_USERS = 20000
NUM_EXPERIMENTS = 5
AVG_SESSIONS_PER_USER = 8
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 3, 31)

print("🚀 Generating A/B Testing Synthetic Data...\n")

# ============================================
# 1. USERS TABLE
# ============================================
print("📊 Generating users table...")

users = pd.DataFrame({
    'user_id': range(1, NUM_USERS + 1),
    'signup_date': [START_DATE + timedelta(days=random.randint(0, 60)) 
                    for _ in range(NUM_USERS)],
    'user_segment': np.random.choice(
        ['new', 'returning', 'VIP'], 
        NUM_USERS, 
        p=[0.40, 0.45, 0.15]
    ),
    'device_type': np.random.choice(
        ['mobile', 'desktop', 'tablet'], 
        NUM_USERS, 
        p=[0.55, 0.35, 0.10]
    ),
    'location': np.random.choice(
        ['United States', 'Canada', 'United Kingdom', 'Germany', 'Australia'], 
        NUM_USERS, 
        p=[0.45, 0.20, 0.15, 0.10, 0.10]
    ),
    'age_group': np.random.choice(
        ['18-25', '26-35', '36-45', '46-55', '56+'], 
        NUM_USERS, 
        p=[0.25, 0.35, 0.25, 0.10, 0.05]
    )
})

users.to_csv('users.csv', index=False)
print(f"✅ Created users.csv ({len(users):,} rows)")

# ============================================
# 2. EXPERIMENTS TABLE
# ============================================
print("📊 Generating experiments table...")

experiments = pd.DataFrame({
    'experiment_id': range(1, NUM_EXPERIMENTS + 1),
    'experiment_name': [
        'checkout_button_color',
        'pricing_display_test',
        'email_subject_line',
        'product_page_layout',
        'free_shipping_threshold'
    ],
    'start_date': [START_DATE] * NUM_EXPERIMENTS,
    'end_date': [END_DATE] * NUM_EXPERIMENTS,
    'hypothesis': [
        'Green button will increase conversions vs blue button',
        'Showing discount percentage will increase purchases',
        'Personalized subject lines will increase email click-through',
        'Grid layout will increase product views vs list layout',
        'Lowering free shipping threshold to $35 will increase order value'
    ],
    'metric_type': ['conversion', 'revenue', 'click_through', 'engagement', 'revenue']
})

experiments.to_csv('experiments.csv', index=False)
print(f"✅ Created experiments.csv ({len(experiments):,} rows)")

# ============================================
# 3. EXPERIMENT ASSIGNMENTS TABLE
# ============================================
print("📊 Generating experiment_assignments table...")

assignments = []
assignment_id = 1

for user_id in range(1, NUM_USERS + 1):
    # Each user participates in 3-4 experiments randomly
    num_experiments_for_user = random.randint(3, 4)
    user_experiments = random.sample(range(1, NUM_EXPERIMENTS + 1), num_experiments_for_user)
    
    for exp_id in user_experiments:
        # 50/50 split between control and treatment
        variant = np.random.choice(['control', 'treatment'], p=[0.5, 0.5])
        assignment_date = users[users['user_id'] == user_id]['signup_date'].values[0]
        
        assignments.append({
            'assignment_id': assignment_id,
            'user_id': user_id,
            'experiment_id': exp_id,
            'variant': variant,
            'assignment_date': assignment_date
        })
        assignment_id += 1

experiment_assignments = pd.DataFrame(assignments)
experiment_assignments.to_csv('experiment_assignments.csv', index=False)
print(f"✅ Created experiment_assignments.csv ({len(experiment_assignments):,} rows)")

# ============================================
# 4. USER SESSIONS TABLE (Main Analysis Table)
# ============================================
print("📊 Generating user_sessions table...")

sessions = []
session_id = 1

# Effect sizes for each experiment (treatment lift)
experiment_effects = {
    1: 0.25,   # 25% lift in conversion for button color
    2: 0.30,   # 30% lift for pricing display
    3: 0.20,   # 20% lift for email subject
    4: 0.15,   # 15% lift for layout
    5: 0.35    # 35% lift for shipping threshold
}

# Base conversion rates by segment
base_conversion_rates = {
    'new': 0.025,
    'returning': 0.040,
    'VIP': 0.080
}

for _, assignment in experiment_assignments.iterrows():
    user_id = assignment['user_id']
    exp_id = assignment['experiment_id']
    variant = assignment['variant']
    
    # Get user segment
    user_segment = users[users['user_id'] == user_id]['user_segment'].values[0]
    base_rate = base_conversion_rates[user_segment]
    
    # Calculate conversion rate based on variant
    if variant == 'treatment':
        conversion_rate = base_rate * (1 + experiment_effects[exp_id])
    else:
        conversion_rate = base_rate
    
    # Generate 7-10 sessions for this user-experiment combo
    num_sessions = random.randint(7, 10)
    
    for _ in range(num_sessions):
        # Random session date between start and end
        days_diff = (END_DATE - START_DATE).days
        session_date = START_DATE + timedelta(days=random.randint(0, days_diff))
        
        # Session behavior
        page_views = random.randint(1, 15)
        time_on_site = random.randint(30, 600)  # 30 sec to 10 min
        
        # Click-through rate (higher page views = higher chance)
        click_prob = min(0.3 + (page_views * 0.02), 0.8)
        clicked_cta = np.random.random() < click_prob
        
        # Conversion (depends on variant and user segment)
        converted = np.random.random() < conversion_rate if clicked_cta else False
        
        # Revenue (if converted)
        if converted:
            # VIP users spend more on average
            if user_segment == 'VIP':
                revenue = np.random.uniform(100, 500)
            elif user_segment == 'returning':
                revenue = np.random.uniform(40, 200)
            else:
                revenue = np.random.uniform(20, 120)
        else:
            revenue = 0.0
        
        sessions.append({
            'session_id': session_id,
            'user_id': user_id,
            'experiment_id': exp_id,
            'session_date': session_date,
            'variant': variant,
            'page_views': page_views,
            'time_on_site': time_on_site,
            'clicked_cta': clicked_cta,
            'converted': converted,
            'revenue': round(revenue, 2)
        })
        session_id += 1

user_sessions = pd.DataFrame(sessions)
user_sessions.to_csv('user_sessions.csv', index=False)
print(f"✅ Created user_sessions.csv ({len(user_sessions):,} rows)")

# ============================================
# 5. DAILY METRICS TABLE (Aggregated)
# ============================================
print("📊 Generating daily_metrics table...")

daily_metrics = user_sessions.groupby(['session_date', 'experiment_id', 'variant']).agg({
    'user_id': 'nunique',
    'converted': ['sum', 'mean'],
    'revenue': ['sum', 'mean']
}).reset_index()

daily_metrics.columns = [
    'date', 'experiment_id', 'variant', 
    'total_users', 'total_conversions', 'conversion_rate',
    'total_revenue', 'avg_revenue_per_user'
]

daily_metrics['conversion_rate'] = daily_metrics['conversion_rate'].round(4)
daily_metrics['total_revenue'] = daily_metrics['total_revenue'].round(2)
daily_metrics['avg_revenue_per_user'] = daily_metrics['avg_revenue_per_user'].round(2)

daily_metrics.to_csv('daily_metrics.csv', index=False)
print(f"✅ Created daily_metrics.csv ({len(daily_metrics):,} rows)")

# ============================================
# SUMMARY STATISTICS
# ============================================
print("\n" + "="*60)
print("📈 DATA GENERATION COMPLETE!")
print("="*60)
print(f"\n📁 Generated Files:")
print(f"   • users.csv ({len(users):,} rows)")
print(f"   • experiments.csv ({len(experiments):,} rows)")
print(f"   • experiment_assignments.csv ({len(experiment_assignments):,} rows)")
print(f"   • user_sessions.csv ({len(user_sessions):,} rows)")
print(f"   • daily_metrics.csv ({len(daily_metrics):,} rows)")

print(f"\n📊 Quick Stats:")
print(f"   • Total Users: {NUM_USERS:,}")
print(f"   • Total Sessions: {len(user_sessions):,}")
print(f"   • Avg Sessions per User: {len(user_sessions) / NUM_USERS:.1f}")
print(f"   • Date Range: {START_DATE.date()} to {END_DATE.date()}")

# Calculate overall conversion rates by variant
print(f"\n🎯 Conversion Rates by Variant (All Experiments):")
for variant in ['control', 'treatment']:
    variant_data = user_sessions[user_sessions['variant'] == variant]
    conv_rate = variant_data['converted'].mean() * 100
    total_revenue = variant_data['revenue'].sum()
    print(f"   • {variant.capitalize()}: {conv_rate:.2f}% conversion | ${total_revenue:,.2f} revenue")

print("\n✨ Data is ready for analysis!")
print("💡 Next steps:")
print("   1. Load CSVs into your analysis environment")
print("   2. Run statistical tests (t-tests, chi-square)")
print("   3. Build Streamlit dashboard for visualization")
print("   4. Calculate confidence intervals and p-values")