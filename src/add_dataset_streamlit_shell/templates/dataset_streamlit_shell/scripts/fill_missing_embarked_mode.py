import json
from datetime import datetime

import pandas as pd

path = r'dataset_streamlit_shell/data/working.csv'
log_path = r'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
mode_value = df['登船港口'].mode(dropna=True).iloc[0]
mask = df['登船港口'].isna()
rows_affected = int(mask.sum())
df.loc[mask, '登船港口'] = mode_value

df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'fill_missing_embarked_mode',
    'columns': ['登船港口'],
    'rows': rows_affected,
    'note': '以登船港口眾數補齊缺失值。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print({'mode_value': mode_value, 'rows_affected': rows_affected})