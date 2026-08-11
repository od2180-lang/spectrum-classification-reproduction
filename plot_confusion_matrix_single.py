#!/usr/bin/env python3
"""Generate single confusion matrix PNG for WITH GATES (72.3% accuracy)."""

import matplotlib.pyplot as plt
import numpy as np

# Labels
techs = ['DAB', 'DVB-T', 'FM', 'GSM', 'LTE', 'TETRA']

# WITH GATES (Phase 4) - from AGENTS.md
with_gates = np.array([
    [49,  5,  0,  0,  2,  0],   # DAB
    [0,  17,  0,  0, 28,  0],   # DVB-T
    [5,  1, 493, 218, 1,  0],   # FM
    [0,  0,  1, 188,  0,  0],   # GSM
    [0,  0,  0,  0,  3,  0],   # LTE
    [0,  0,  0,  0,  0,  2],   # TETRA
])

# Normalize rows to decimals (0-1)
def normalize_rows(mat):
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums

with_norm = normalize_rows(with_gates)

# Create figure (square)
fig, ax = plt.subplots(figsize=(10, 10))
fig.suptitle('Confusion Matrix — WITH Gates (Phase 4)', fontsize=26, fontweight='bold')

# Plot WITH GATES
ax.imshow(with_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')
ax.set_title('Accuracy: 72.3% | Classified: 1,040 / 1,916 (54.3%)', fontsize=18, fontweight='bold', pad=15)
ax.set_xticks(range(len(techs)))
ax.set_xticklabels(techs, fontsize=18, fontweight='normal')
ax.set_yticks(range(len(techs)))
ax.set_yticklabels(techs, fontsize=18, fontweight='normal')
ax.set_xlabel('Predicted Label', fontsize=20, fontweight='bold', labelpad=10)
ax.set_ylabel('True Label', fontsize=20, fontweight='bold', labelpad=10)

# Add grid lines (light gray, thin)
ax.set_xticks(np.arange(-0.5, len(techs), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(techs), 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', size=0)

# Keep aspect ratio equal
ax.set_aspect('equal')

# Annotate with_gates - white on dark, black on light, decimal format
for i in range(len(techs)):
    for j in range(len(techs)):
        val = with_gates[i, j]
        dec = with_norm[i, j]
        if val > 0:
            color = 'white' if dec > 0.5 else 'black'
            ax.text(j, i, f'{dec:.2f}', ha='center', va='center', fontsize=20, color=color, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/jovyan/work/project/confusion_matrix_72.png', dpi=200, bbox_inches='tight')
print("Saved to confusion_matrix_72.png")
