import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'

labels = ['DAB', 'DVBT', 'FM', 'GSM', 'LTE', 'TETRA']
cm = np.array([
    [0.33, 0.33, 0.13, 0.13, 0.08, 0.00],
    [0.02, 0.95, 0.02, 0.00, 0.01, 0.01],
    [0.00, 0.02, 0.96, 0.01, 0.00, 0.01],
    [0.00, 0.06, 0.18, 0.76, 0.00, 0.00],
    [0.02, 0.05, 0.04, 0.04, 0.85, 0.00],
    [0.00, 0.05, 0.36, 0.00, 0.00, 0.59],
])

cm_pct = cm * 100

fig, ax = plt.subplots(figsize=(4, 3.5))

im = ax.imshow(cm_pct, cmap='PuRd', vmin=0, vmax=100)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('')
ax.set_ylabel('')

for i in range(len(labels)):
    for j in range(len(labels)):
        val = cm_pct[i, j]
        color = 'white' if val > 50 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=13, color=color, fontweight='normal')

plt.tight_layout()
plt.savefig('/home/jovyan/work/project/confusion_matrix_author.png', dpi=150, facecolor='white')
print('Saved: confusion_matrix_author.png')
