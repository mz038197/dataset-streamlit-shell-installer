import json
from datetime import datetime

import pandas as pd

path = 'dataset_streamlit_shell/data/working.csv'
log_path = 'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
cols = [c for c in df.columns if c != 'PassengerId']
dup_mask = df.duplicated(subset=cols, keep='first')
removed = df.loc[dup_mask].copy()
df2 = df.loc[~dup_mask].copy()
df2.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'drop_duplicate_rows',
    'columns': cols,
    'rows': int(dup_mask.sum()),
    'note': '刪除排除 PassengerId 後判定為重複的資料列。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print({'removed_rows': int(dup_mask.sum()), 'removed_passenger_ids': removed['PassengerId'].tolist()})
