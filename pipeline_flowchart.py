import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'serif'

def create_pipeline(stages, colors, filename):
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')

    box_width = 1.8
    box_height = 1.0
    gap = 0.6
    start_x = 0.5
    y_center = 1.0

    for i, (stage, color) in enumerate(zip(stages, colors)):
        x = start_x + i * (box_width + gap)
        y = y_center - box_height / 2
        
        rect = mpatches.FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2,
            alpha=0.85
        )
        ax.add_patch(rect)
        
        ax.text(x + box_width / 2, y_center, stage,
                ha='center', va='center',
                fontsize=18, fontweight='bold',
                color='white')
        
        if i < len(stages) - 1:
            arrow_x = x + box_width + 0.05
            arrow_end_x = x + box_width + gap - 0.05
            ax.annotate('', xy=(arrow_end_x, y_center),
                        xytext=(arrow_x, y_center),
                        arrowprops=dict(arrowstyle='->', color='#333333', lw=3))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(filename.replace('.png', '.svg'), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()

base_color = '#009b8a'
highlight_color = '#fb0f78'

# Original (base)
stages = ["Raw Data", "Preprocessing", "Autoencoder", "LSTM\nClassifier"]
colors_original = [base_color, base_color, base_color, base_color]
create_pipeline(stages, colors_original, 'pipeline_flowchart.png')

# Preprocessing highlighted
colors_preproc = [base_color, highlight_color, base_color, base_color]
create_pipeline(stages, colors_preproc, 'pipeline_flowchart_preprocessing.png')

# All ML stages highlighted
colors_all_ml = [base_color, highlight_color, highlight_color, highlight_color]
create_pipeline(stages, colors_all_ml, 'pipeline_flowchart_all_ml.png')

print("Saved:")
print("  - pipeline_flowchart.png (original)")
print("  - pipeline_flowchart_preprocessing.png")
print("  - pipeline_flowchart_all_ml.png")
