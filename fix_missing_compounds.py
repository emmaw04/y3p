import pandas as pd
import numpy as np

# Load datasets
print("Loading datasets...")
df1 = pd.read_csv('data/processed/dataset1.csv')
df2 = pd.read_csv('data/processed/dataset2.csv')

missing_before_1 = df1['current_compound'].isna().sum()
missing_before_2 = df2['current_compound'].isna().sum()

print(f"Missing in dataset1 before: {missing_before_1}")
print(f"Missing in dataset2 before: {missing_before_2}")

# Sort values to ensure chronological order
df1 = df1.sort_values(['race_id', 'driver_id', 'lapno']).reset_index(drop=True)

# Create stint_id
# A stint changes if the previous lap had y_pit == 1
df1['pitted_prev'] = df1.groupby(['race_id', 'driver_id'])['y_pit'].shift(1).fillna(0)
df1['stint_id'] = df1.groupby(['race_id', 'driver_id'])['pitted_prev'].cumsum()

# Forward and backward fill within stints
df1['current_compound'] = df1.groupby(['race_id', 'driver_id', 'stint_id'])['current_compound'].ffill()
df1['current_compound'] = df1.groupby(['race_id', 'driver_id', 'stint_id'])['current_compound'].bfill()

missing_after_1 = df1['current_compound'].isna().sum()
print(f"Missing in dataset1 after: {missing_after_1}")

# Clean up temporary columns
df1 = df1.drop(columns=['pitted_prev', 'stint_id'])

# Update dataset2
df1_compounds = df1.set_index(['race_id', 'driver_id', 'lapno'])['current_compound']
df2 = df2.set_index(['race_id', 'driver_id', 'lapno'])
df2['current_compound'] = df2['current_compound'].fillna(df1_compounds)
df2 = df2.reset_index()

# Reorder columns to match original dataset2
df2_cols = pd.read_csv('data/processed/dataset2.csv', nrows=0).columns
df2 = df2[df2_cols]

missing_after_2 = df2['current_compound'].isna().sum()
print(f"Missing in dataset2 after: {missing_after_2}")

# Save
print("Saving datasets...")
df1.to_csv('data/processed/dataset1.csv', index=False)
df2.to_csv('data/processed/dataset2.csv', index=False)
print("Done.")
