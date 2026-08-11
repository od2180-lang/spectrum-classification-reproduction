import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Confusion matrix data (rows=actual, columns=predicted)
labels = ['DAB', 'DVB-T', 'FM', 'GSM', 'LTE', 'TETRA']
cm = np.array([
    [49,  5,   0,   0,  2,  0],   # DAB
    [ 0, 17,   0,   0, 28,  0],   # DVB-T
    [ 5,  1, 493, 218,  1,  0],   # FM
    [ 0,  0,   1, 188,  0,  0],   # GSM
    [ 0,  0,   0,   0,  3,  0],   # LTE
    [ 0,  0,   0,   0,  0,  2],   # TETRA
], dtype=float)

# Convert to percentages (row-normalized)
row_sums = cm.sum(axis=1, keepdims=True)
cm_pct = (cm / row_sums) * 100

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_pct, cmap='PuRd', vmin=0, vmax=100)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix — 72.3% Accuracy', fontsize=13, fontweight='bold')

# Add text annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        val = cm_pct[i, j]
        color = 'white' if val > 50 else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                fontsize=10, color=color, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('% of Actual Class', fontsize=10)

plt.tight_layout()
plt.savefig('/home/jovyan/work/project/confusion_matrix_72.png', dpi=150, facecolor='#fb0f78')
print('Saved: confusion_matrix_72.png')
