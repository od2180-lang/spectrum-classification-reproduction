import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'

labels = ['DAB', 'DVB-T', 'FM', 'GSM', 'LTE', 'TETRA']
cm = np.array([
    [ 72,  16,  19, 171,   9,   0],
    [ 36,  38,   3, 253,  43,   0],
    [  7,   1, 493, 220,   3,   0],
    [  7,   2,   3, 300,   2,   0],
    [ 10,   3,   0,  31,  15,   0],
    [  1,   0, 121,  30,   1,   2],
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
                fontsize=13, color=color, fontweight='normal')

plt.tight_layout()
plt.savefig('/home/jovyan/work/project/confusion_matrix_no_gates.png', dpi=150, facecolor='white')
print('Saved: confusion_matrix_no_gates.png')
