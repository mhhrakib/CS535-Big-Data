import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1) Read metrics from CSVs
models = ['pegasus', 'bart', 'led']
paths = {m: f"outputs/{m}/best_model/eval_metrics.csv" for m in models}

records = []
for m in models:
    df = pd.read_csv(paths[m])
    df['model'] = m.title().upper()
    records.append(df)

metrics_df = pd.concat(records, ignore_index=True).set_index('model')

# 2) Prepare data
rouge1 = metrics_df['rouge1']
rouge2 = metrics_df['rouge2']
rougeL = metrics_df['rougeL']
x = np.arange(len(models))
width = 0.25

# 3) Plot grouped bar chart with annotations
fig, ax = plt.subplots()
bars1 = ax.bar(x - width, rouge1, width)
bars2 = ax.bar(x,       rouge2, width)
bars3 = ax.bar(x + width, rougeL, width)

ax.set_xticks(x)
ax.set_xticklabels(metrics_df.index)
ax.set_ylabel('Score')
ax.set_title('ROUGE-1, ROUGE-2, ROUGE-L by Model')
ax.legend(['ROUGE-1','ROUGE-2','ROUGE-L'])

# Annotate each bar with its value (2 decimal places)
for bar_group in (bars1, bars2, bars3):
    for bar in bar_group:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=8
        )

plt.tight_layout()

# 4) Save figure
os.makedirs("outputs/plots", exist_ok=True)
fig.savefig("outputs/plots/rouge_metrics_labeled.png")
plt.close()

