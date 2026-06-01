import pandas as pd

path = 'dataset_streamlit_shell/data/working.csv'
df = pd.read_csv(path)

for c in ['兄弟姊妹/配偶人數', '父母/子女人數']:
    s = df[c]
    print(c)
    print('dtype=', s.dtype)
    print('nunique=', s.nunique())
    print(s.describe().to_string())
    print()
