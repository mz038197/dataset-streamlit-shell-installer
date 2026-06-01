import json
from datetime import datetime

import pandas as pd

path = r'dataset_streamlit_shell/data/working.csv'
log_path = r'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
before_rows = len(df)

df = df.drop_duplicates(subset=['姓名'], keep='first')
after_rows = len(df)
removed_rows = before_rows - after_rows

df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'drop_duplicates_by_name',
    'columns': ['姓名'],
    'rows': int(removed_rows),
    'note': '依姓名欄位刪除重複資料，保留第一筆出現的紀錄。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print({'before_rows': before_rows, 'after_rows': after_rows, 'removed_rows': removed_rows})