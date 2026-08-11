import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'

labels = ['DAB', 'DVB-T', 'FM', 'GSM', 'LTE', 'TETRA']
cm = np.array([
    [49,  5,   0,   0,  2,  0],
    [ 0, 17,   0,   0, 28,  0],
    [ 5,  1, 493, 218,  1,  0],
    [ 0,  0,   1, 188,  0,  0],
    [ 0,  0,   0,   0,  3,  0],
    [ 0,  0,   0,   0,  0,  2],
], dtype=float)

row_sums = cm.sum(axis=1, keepdims=True)
cm_pct = (cm / row_sums) * 100

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
                fontsize=10, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/jovyan/work/project/confusion_matrix_72_v2.png', dpi=150, facecolor='white')
print('Saved: confusion_matrix_72_v2.png')
