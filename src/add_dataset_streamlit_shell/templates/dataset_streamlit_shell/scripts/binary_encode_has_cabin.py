import json
from datetime import datetime

import pandas as pd

path = 'dataset_streamlit_shell/data/working.csv'
log_path = 'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
df['是否有艙房'] = df['是否有艙房'].map({'有': 1, '無': 0})
df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'binary_encode_has_cabin',
    'columns': ['是否有艙房'],
    'rows': int(len(df)),
    'note': '將是否有艙房欄位轉為有=1、無=0。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(df['是否有艙房'].head().to_string(index=False))
