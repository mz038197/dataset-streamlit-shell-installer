import json
from datetime import datetime

import pandas as pd

path = 'dataset_streamlit_shell/data/working.csv'
log_path = 'dataset_streamlit_shell/data/cleaning_log.jsonl'

mapping = {3: 1, 2: 2, 1: 3}

df = pd.read_csv(path)
df['艙等'] = df['艙等'].map(mapping)
df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'recode_class_reverse',
    'columns': ['艙等'],
    'rows': int(len(df)),
    'note': '將艙等依 3=1、2=2、1=3 重新編碼。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(df['艙等'].value_counts().sort_index().to_string())
