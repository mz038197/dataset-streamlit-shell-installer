import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

base = Path('dataset_streamlit_shell')
input_path = base / 'data' / 'working.csv'
out_path = base / 'data' / 'average_radar_chart.png'

# 讀取資料

df = pd.read_csv(input_path)
cols = ['Age', 'Fare', 'SibSp', 'Parch']
summary = df[cols].mean()

# 雷達圖設定
labels = cols
values = summary.tolist()
values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, values, linewidth=2, label='平均值')
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_title('Age / Fare / SibSp / Parch 平均值雷達圖', fontsize=16, pad=20)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(out_path)
