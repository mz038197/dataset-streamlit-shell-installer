import pandas as pd

path = 'dataset_streamlit_shell/data/working.csv'
df = pd.read_csv(path)
num_cols = ['是否生還', '艙等', '年齡', '兄弟姊妹/配偶人數', '父母/子女人數', '票價']

for c in num_cols:
    s = df[c].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    n = ((s < low) | (s > high)).sum()
    print(f'{c}\toutliers={n}\tlow={low}\thigh={high}')
