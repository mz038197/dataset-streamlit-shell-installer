import json
from datetime import datetime

import pandas as pd

path = 'dataset_streamlit_shell/data/working.csv'
log_path = 'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
dummies = pd.get_dummies(df['性別'], prefix='性別')
df = pd.concat([df.drop(columns=['性別']), dummies], axis=1)
df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'one_hot_encode_sex',
    'columns': ['性別', '性別_female', '性別_male'],
    'rows': int(len(df)),
    'note': '將性別欄位進行 one-hot encoding。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(df[['性別_female', '性別_male']].head().to_string(index=False))
