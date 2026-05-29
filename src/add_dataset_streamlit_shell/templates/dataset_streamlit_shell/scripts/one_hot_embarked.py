import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_PATH = Path('dataset_streamlit_shell/data/working.csv')
LOG_PATH = Path('dataset_streamlit_shell/data/cleaning_log.jsonl')

df = pd.read_csv(DATA_PATH)

embarked_dummies = pd.get_dummies(df['登船港口'], prefix='登船港口')
df = pd.concat([df.drop(columns=['登船港口']), embarked_dummies], axis=1)

base_cols = [c for c in df.columns if c not in ['登船港口_C', '登船港口_Q', '登船港口_S']]
if '票號' in base_cols:
    insert_at = base_cols.index('票號') + 1
else:
    insert_at = len(base_cols)
new_cols = base_cols[:insert_at] + ['登船港口_C', '登船港口_Q', '登船港口_S'] + base_cols[insert_at:]
df = df[new_cols]

df.to_csv(DATA_PATH, index=False, encoding='utf-8-sig')

entry = {
    'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    'actor': 'agent',
    'action': 'one_hot_encode_embarked',
    'columns': ['登船港口'],
    'rows': int(len(df)),
    'note': '將登船港口欄位轉為 One-Hot 編碼欄位。'
}
with LOG_PATH.open('a', encoding='utf-8') as f:
    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(df[['登船港口_C', '登船港口_Q', '登船港口_S']].head())
print(df.shape)
