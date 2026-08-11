#!/usr/bin/env python3
"""Generate clean spectrogram with time on x-axis, frequency on y-axis."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

np.random.seed(42)

n_time = 200
n_freq = 214

# Noise
data = np.random.normal(-72, 3, (n_freq, n_time))

# Single signal (vertical stripe)
signal_center = 107
signal_width = 5
for t in range(n_time):
    for f in range(signal_center - signal_width, signal_center + signal_width + 1):
        dist = abs(f - signal_center) / signal_width
        data[f, t] += 25 * np.exp(-2 * dist**2)
    data[signal_center - 2:signal_center + 3, t] += np.random.normal(0, 1, 5)

data = gaussian_filter(data, sigma=0.8)

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('white')

ax.imshow(data, aspect='auto', cmap='jet', vmin=-80, vmax=-20, origin='lower')

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)

plt.tight_layout()
plt.savefig('/home/jovyan/work/project/spec_author.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================
# OURS: multiple signals
# ============================================================
np.random.seed(123)
n_time = 200
n_freq = 1200

data2 = np.random.normal(-72, 4, (n_freq, n_time))
for f in range(n_freq):
    data2[f, :] += 5 * np.sin(2 * np.pi * f / 400)

signals = [
    (50, 80, 20), (150, 190, 22), (300, 340, 18),
    (450, 490, 21), (600, 640, 19), (100, 110, 15),
    (250, 260, 14), (400, 410, 16), (520, 555, 20),
    (700, 735, 18), (800, 1000, 25), (1050, 1150, 28),
]

for start, end, strength in signals:
    width = end - start
    center = (start + end) // 2
    for t in range(n_time):
        for f in range(start, end):
            dist = abs(f - center) / (width / 2)
            if width > 100:
                power = strength * (0.8 + 0.2 * np.sin(2 * np.pi * f / 50))
            else:
                power = strength * np.exp(-3 * dist**2)
            data2[f, t] += power + np.random.normal(0, 1)

data2 = gaussian_filter(data2, sigma=0.5)

fig2, ax2 = plt.subplots(figsize=(14, 5))
fig2.patch.set_facecolor('white')

ax2.imshow(data2, aspect='auto', cmap='jet', vmin=-80, vmax=-10, origin='lower')

ax2.set_xticks([])
ax2.set_yticks([])

for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)

plt.tight_layout()
plt.savefig('/home/jovyan/work/project/spec_ours.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Done: spec_author.png, spec_ours.png")
