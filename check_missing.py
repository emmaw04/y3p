import pandas as pd
df2 = pd.read_csv('data/processed/dataset2.csv')
missing = df2[df2['current_compound'].isna()]
print(missing[['race_id', 'driver_id', 'lapno', 'y_compound', 'current_compound']])
