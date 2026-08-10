#!/usr/bin/env python3
"""Generate side-by-side confusion matrix PNG."""

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

# WITHOUT GATES (from run_no_gates.py output)
without_gates = np.array([
    [72,  16,  19, 171,  9,  0],  # DAB
    [36,  38,   3, 253, 43,  0],  # DVB-T
    [7,   1, 493, 220,  3,  0],   # FM
    [7,   2,   3, 300,  2,  0],   # GSM
    [10,  3,   0,  31, 15,  0],   # LTE
    [1,   0, 121,  30,  1,  2],   # TETRA
])

# Normalize rows to percentages
def normalize_rows(mat):
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums * 100

with_norm = normalize_rows(with_gates)
without_norm = normalize_rows(without_gates)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('Confusion Matrix Comparison: With vs Without Width/Entropy Gates', fontsize=22, fontweight='bold')

# Plot WITH GATES
axes[0].imshow(with_norm, cmap='Greys', vmin=0, vmax=100, aspect='auto')
axes[0].set_title('WITH Gates (Phase 4)\nAccuracy: 72.3% | Classified: 1,040/1,916 (54.3%)', fontsize=16, fontweight='bold')
axes[0].set_xticks(range(len(techs)))
axes[0].set_xticklabels(techs, fontsize=14, fontweight='bold')
axes[0].set_yticks(range(len(techs)))
axes[0].set_yticklabels(techs, fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Expected', fontsize=16, fontweight='bold')

# Annotate with_gates
for i in range(len(techs)):
    for j in range(len(techs)):
        val = with_gates[i, j]
        pct = with_norm[i, j]
        if val > 0:
            color = 'white' if pct > 50 else 'black'
            axes[0].text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center', fontsize=14, color=color, fontweight='bold')

# Plot WITHOUT GATES
axes[1].imshow(without_norm, cmap='Greys', vmin=0, vmax=100, aspect='auto')
axes[1].set_title('WITHOUT Gates (True Generalization)\nAccuracy: 47.3% | Classified: 1,916/1,916 (100%)', fontsize=16, fontweight='bold')
axes[1].set_xticks(range(len(techs)))
axes[1].set_xticklabels(techs, fontsize=14, fontweight='bold')
axes[1].set_yticks(range(len(techs)))
axes[1].set_yticklabels(techs, fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=16, fontweight='bold')
axes[1].set_ylabel('Expected', fontsize=16, fontweight='bold')

# Annotate without_gates
for i in range(len(techs)):
    for j in range(len(techs)):
        val = without_gates[i, j]
        pct = without_norm[i, j]
        if val > 0:
            color = 'white' if pct > 50 else 'black'
            axes[1].text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center', fontsize=14, color=color, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/jovyan/work/project/confusion_matrix_comparison.png', dpi=200, bbox_inches='tight')
print("Saved to confusion_matrix_comparison.png")