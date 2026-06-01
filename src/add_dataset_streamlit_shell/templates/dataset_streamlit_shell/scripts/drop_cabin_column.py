import json
from datetime import datetime

import pandas as pd

path = r'dataset_streamlit_shell/data/working.csv'
log_path = r'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
if '艙房' in df.columns:
    df = df.drop(columns=['艙房'])
    removed = 1
else:
    removed = 0

df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'drop_cabin_column',
    'columns': ['艙房'],
    'rows': int(len(df)),
    'note': '刪除艙房欄位並保留其他資料欄位。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print({'removed_column': bool(removed), 'remaining_columns': df.columns.tolist()})