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

fig, ax = plt.subplots(figsize=(8, 6))

# White background, grid lines as separators
ax.set_facecolor('white')
ax.imshow(np.ones_like(cm_pct), cmap='Greys', vmin=0.9, vmax=1.1)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix — 72.3% Accuracy', fontsize=13, fontweight='bold')

# Add text annotations (no %, 1 decimal)
text_color = '#fb0f78'
for i in range(len(labels)):
    for j in range(len(labels)):
        val = cm_pct[i, j]
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=10, color=text_color, fontweight='bold')

# Draw grid
for i in range(len(labels) + 1):
    ax.axhline(i - 0.5, color='grey', linewidth=0.5)
    ax.axvline(i - 0.5, color='grey', linewidth=0.5)

ax.tick_params(top=False, bottom=False, left=False, right=False)
plt.tight_layout()
plt.savefig('/home/jovyan/work/project/confusion_matrix_72_clean.png', dpi=150, facecolor='white')
print('Saved: confusion_matrix_72_clean.png')
