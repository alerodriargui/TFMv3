"""Generate high-quality architecture diagram for the ConvAutoencoder."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors
encoder_color = '#2980b9'
decoder_color = '#c0392b'
bottleneck_color = '#8e44ad'
input_color = '#95a5a6'
arrow_color = '#2c3e50'

# ============ ENCODER ============
encoder_blocks = [
    (1.5, 5.0, 'Input\n1$\\times$240$\\times$240', input_color, 1.4),
    (3.5, 5.0, 'Conv2d 3$\\times$3, stride=2\n8 channels\n120$\\times$120', encoder_color, 1.4),
    (5.5, 5.0, 'Conv2d 3$\\times$3, stride=2\n16 channels\n60$\\times$60', encoder_color, 1.4),
    (7.5, 5.0, 'Conv2d 3$\\times$3, stride=2\n32 channels\n30$\\times$30', encoder_color, 1.4),
]

# ============ BOTTLENECK ============
bottleneck = (9.0, 5.0, 'Bottleneck\n32$\\times$30$\\times$30', bottleneck_color, 1.0)

# ============ DECODER ============
decoder_blocks = [
    (10.5, 5.0, 'ConvTranspose2d 4$\\times$4\n16 channels\n60$\\times$60', decoder_color, 1.4),
    (12.5, 5.0, 'ConvTranspose2d 4$\\times$4\n8 channels\n120$\\times$120', decoder_color, 1.4),
    (14.5, 5.0, 'ConvTranspose2d 4$\\times$4\n1 channel\n240$\\times$240\n+ Sigmoid', decoder_color, 1.4),
]

# Draw encoder blocks
for i, (x, y, label, color, width) in enumerate(encoder_blocks):
    # Box
    box = FancyBboxPatch((x - width/2, y - 0.6), width, 1.2,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='black', linewidth=2,
                          alpha=0.85 if color != input_color else 0.3)
    ax.add_patch(box)
    # Label
    text_color = 'black' if color == input_color else 'white'
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=text_color)
    # Arrow to next
    if i < len(encoder_blocks) - 1:
        arrow = FancyArrowPatch((x + width/2 + 0.05, y), 
                               (encoder_blocks[i+1][0] - encoder_blocks[i+1][4]/2 - 0.05, y),
                               arrowstyle='->', mutation_scale=20,
                               color=arrow_color, linewidth=2.5)
        ax.add_patch(arrow)

# Arrow from encoder to bottleneck
arrow = FancyArrowPatch((7.5 + 0.75, y), (9.0 - 0.55, y),
                       arrowstyle='->', mutation_scale=20,
                       color=bottleneck_color, linewidth=2.5, linestyle='--')
ax.add_patch(arrow)

# Bottleneck box
box = FancyBboxPatch((9.0 - 0.5, 5.0 - 0.6), 1.0, 1.2,
                      boxstyle="round,pad=0.08",
                      facecolor=bottleneck_color, edgecolor='black', linewidth=2.5,
                      alpha=0.9)
ax.add_patch(box)
ax.text(9.0, 5.0, 'Bottleneck\n32$\\times$30$\\times$30', ha='center', va='center', 
        fontsize=9, fontweight='bold', color='white')

# Arrow from bottleneck to decoder
arrow = FancyArrowPatch((9.0 + 0.55, y), (10.5 - 0.75, y),
                       arrowstyle='->', mutation_scale=20,
                       color=arrow_color, linewidth=2.5)
ax.add_patch(arrow)

# Draw decoder blocks
for i, (x, y, label, color, width) in enumerate(decoder_blocks):
    box = FancyBboxPatch((x - width/2, y - 0.6), width, 1.2,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='black', linewidth=2,
                          alpha=0.85)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')
    if i < len(decoder_blocks) - 1:
        arrow = FancyArrowPatch((x + width/2 + 0.05, y), 
                               (decoder_blocks[i+1][0] - decoder_blocks[i+1][4]/2 - 0.05, y),
                               arrowstyle='->', mutation_scale=20,
                               color=arrow_color, linewidth=2.5)
        ax.add_patch(arrow)

# Section labels
ax.text(4.5, 8.5, 'ENCODER', ha='center', va='center',
        fontsize=18, fontweight='bold', color=encoder_color)
ax.text(12.5, 8.5, 'DECODER', ha='center', va='center',
        fontsize=18, fontweight='bold', color=decoder_color)

# Activation info
ax.text(4.5, 2.5, 'ReLU activation after each convolution', ha='center', fontsize=11,
        fontstyle='italic', color='#7f8c8d')
ax.text(4.5, 1.8, 'Loss: MAE (L1)', ha='center', fontsize=11,
        fontstyle='italic', color='#7f8c8d')
ax.text(4.5, 1.1, 'Total parameters: 16,281', ha='center', fontsize=11,
        fontstyle='italic', color='#7f8c8d')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=encoder_color, label='Encoder (Conv2d, stride=2)'),
    mpatches.Patch(facecolor=bottleneck_color, label='Bottleneck'),
    mpatches.Patch(facecolor=decoder_color, label='Decoder (ConvTranspose2d)'),
]
ax.legend(handles=legend_elements, loc='lower right', ncol=1, fontsize=11,
          framealpha=0.9, edgecolor='gray')

plt.tight_layout()
plt.savefig(r'C:\Users\Alex\OneDrive\Documentos\GitHub\TFMv3\docs\assets\ae_architecture.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Architecture diagram saved at 200 DPI!")
