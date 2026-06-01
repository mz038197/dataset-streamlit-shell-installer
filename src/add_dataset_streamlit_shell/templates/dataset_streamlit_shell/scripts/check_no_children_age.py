import pandas as pd

path = r'dataset_streamlit_shell/data/working.csv'
df = pd.read_csv(path)

subset = df[df['父母/子女人數'] == 0]
print('rows_no_children=', len(subset))
print('age_mean=', subset['年齡'].mean())
print('age_median=', subset['年齡'].median())
print('age_min=', subset['年齡'].min())
print('age_max=', subset['年齡'].max())
print('\nAge describe:')
print(subset['年齡'].describe().to_string())