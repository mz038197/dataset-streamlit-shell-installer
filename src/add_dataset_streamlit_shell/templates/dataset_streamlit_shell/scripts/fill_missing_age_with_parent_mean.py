import json
from datetime import datetime

import pandas as pd

path = r'dataset_streamlit_shell/data/working.csv'
log_path = r'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
mask = (df['父母/子女人數'] > 0) & (df['年齡'].isna())
fill_value = 23.006062176165802
rows_affected = int(mask.sum())
df.loc[mask, '年齡'] = fill_value

df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'fill_missing_age_with_parent_mean',
    'columns': ['年齡'],
    'rows': rows_affected,
    'note': '以有子女族群的平均年齡補齊年齡欄位缺失值。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print({'fill_value': fill_value, 'rows_affected': rows_affected})