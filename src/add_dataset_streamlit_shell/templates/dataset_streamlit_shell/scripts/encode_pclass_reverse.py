import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_PATH = Path('dataset_streamlit_shell/data/working.csv')
LOG_PATH = Path('dataset_streamlit_shell/data/cleaning_log.jsonl')

df = pd.read_csv(DATA_PATH)

mapping = {1: 3, 2: 2, 3: 1}
df['艙等'] = df['艙等'].map(mapping)

df.to_csv(DATA_PATH, index=False, encoding='utf-8-sig')

entry = {
    'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    'actor': 'agent',
    'action': 'reverse_encode_pclass',
    'columns': ['艙等'],
    'rows': int(len(df)),
    'note': '將艙等欄位反向編碼為數值越大代表等級越高。'
}
with LOG_PATH.open('a', encoding='utf-8') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(df['艙等'].head())
print(df['艙等'].value_counts().sort_index())
