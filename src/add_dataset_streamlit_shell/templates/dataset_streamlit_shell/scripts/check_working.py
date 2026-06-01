import pandas as pd

path = r'dataset_streamlit_shell/data/working.csv'
df = pd.read_csv(path)
print(df.shape)
print(df.head(10).to_string(index=False))
print('\nDUPLICATES_FULL_ROWS=', df.duplicated().sum())
print('\nMissing by column:')
print(df.isna().sum().to_string())
