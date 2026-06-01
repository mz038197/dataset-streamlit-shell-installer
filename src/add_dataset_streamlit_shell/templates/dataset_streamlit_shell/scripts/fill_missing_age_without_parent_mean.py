import json
from datetime import datetime

import pandas as pd

path = 'dataset_streamlit_shell/data/working.csv'
log_path = 'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
mask = df['年齡'].isna() & (df['父母/子女人數'] == 0)
fill_value = df.loc[df['父母/子女人數'] == 0, '年齡'].mean()
df.loc[mask, '年齡'] = fill_value
df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'fill_missing_age_without_parent_mean',
    'columns': ['年齡'],
    'rows': int(mask.sum()),
    'note': '以沒有父母／子女人數者的平均年齡補齊其年齡空值。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print({'filled_rows': int(mask.sum()), 'fill_value': fill_value})
