import json
from datetime import datetime

import pandas as pd

path = 'dataset_streamlit_shell/data/working.csv'
log_path = 'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
df['是否有艙房'] = df['艙房'].notna().map({True: '有', False: '無'})
df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'add_has_cabin_column',
    'columns': ['艙房', '是否有艙房'],
    'rows': int(len(df)),
    'note': '根據艙房是否缺失新增是否有艙房欄位。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(df[['艙房', '是否有艙房']].head(10).to_string(index=False))
