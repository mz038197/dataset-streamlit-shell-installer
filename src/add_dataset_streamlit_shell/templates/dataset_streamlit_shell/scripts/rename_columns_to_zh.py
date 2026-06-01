import json
from datetime import datetime

import pandas as pd

path = 'dataset_streamlit_shell/data/working.csv'
log_path = 'dataset_streamlit_shell/data/cleaning_log.jsonl'

df = pd.read_csv(path)
rename_map = {
    'PassengerId': '乘客編號',
    'Survived': '是否生還',
    'Pclass': '艙等',
    'Name': '姓名',
    'Sex': '性別',
    'Age': '年齡',
    'SibSp': '兄弟姊妹/配偶人數',
    'Parch': '父母/子女人數',
    'Ticket': '票號',
    'Fare': '票價',
    'Cabin': '艙房',
    'Embarked': '登船港口',
}
df = df.rename(columns=rename_map)
df.to_csv(path, index=False)

record = {
    'created_at': datetime.now().replace(microsecond=0).isoformat(),
    'actor': 'agent',
    'action': 'rename_columns_zh',
    'columns': list(rename_map.values()),
    'rows': int(len(df)),
    'note': '將工作資料的欄位名稱改為繁體中文。',
}
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(df.columns.tolist())
