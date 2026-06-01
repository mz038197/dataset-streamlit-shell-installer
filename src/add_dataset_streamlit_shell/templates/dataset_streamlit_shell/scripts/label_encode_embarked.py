import json
from datetime import datetime

import pandas as pd

path = r'dataset_streamlit_shell/data/working.csv'
log_path = r'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
unique_values = sorted(df['登船港口'].dropna().unique().tolist())
mapping = {value: idx for idx, value in enumerate(unique_values)}
df['登船港口'] = df['登船港口'].map(mapping)
df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'label_encode_embarked',
    'columns': ['登船港口'],
    'rows': int(len(df)),
    'note': '將登船港口以 Label Encoding 轉為數值編碼。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print({'mapping': mapping})