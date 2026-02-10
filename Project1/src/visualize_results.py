"""
Visualization of Prototype Selection Experiment Results

Generates publication-quality plots with modern, visually appealing styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Modern color palette (inspired by Tableau/D3)
COLORS = {
    'ours': '#1f77b4',       # Blue
    'boundary': '#2ca02c',   # Green
    'random': '#d62728',     # Red
    'accent': '#ff7f0e',     # Orange
    'gray': '#7f7f7f',
    'light_gray': '#c7c7c7',
    'bg': '#fafafa',
}

# Set modern style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
})

# Experimental results
RESULTS = {
    'M_values': [10000, 5000, 1000],
    'accuracy': {
        'variance_centroid': [0.9536, 0.9458, 0.9247],
        'cluster_boundary': [0.9534, 0.9456, 0.9246],
        'random': [0.9476, 0.9363, 0.8856],
    },
    'ci_95': {
        'variance_centroid': [0.0011, 0.0018, 0.0015],
        'cluster_boundary': [0.0011, 0.0020, 0.0016],
        'random': [0.0008, 0.0008, 0.0045],
    },
    'time_select': {
        'variance_centroid': [234.27, 51.37, 15.60],
        'cluster_boundary': [267.23, 60.37, 14.11],
        'random': [0.0, 0.0, 0.0],
    },
    'per_class_accuracy': {
        1000: {
            'variance_centroid': {0: 0.9829, 1: 0.9940, 2: 0.9095, 3: 0.9091, 4: 0.8943,
                                   5: 0.9195, 6: 0.9666, 7: 0.9204, 8: 0.8526, 9: 0.8894},
            'random': {0: 0.9671, 1: 0.9937, 2: 0.8200, 3: 0.8820, 4: 0.8485,
                       5: 0.8372, 6: 0.9511, 7: 0.8967, 8: 0.7938, 9: 0.8492},
        }
    },
    'prototype_distribution': {
        1000: {
            'variance_centroid': {0: 118, 1: 53, 2: 121, 3: 107, 4: 97,
                                   5: 113, 6: 102, 7: 89, 8: 108, 9: 92},
            'random': {0: 95, 1: 134, 2: 96, 3: 102, 4: 98,
                       5: 78, 6: 97, 7: 108, 8: 96, 9: 96},
        }
    }
}

# Extended results including quick tests
ALL_M = [10000, 5000, 1000, 500, 100]
ALL_ACC = {
    'variance_centroid': [0.9536, 0.9458, 0.9247, 0.9140, 0.8619],
    'cluster_boundary': [0.9534, 0.9456, 0.9246, 0.9138, 0.8632],
    'random': [0.9476, 0.9363, 0.8856, 0.8470, 0.7239],
}


def plot_main_comparison():
    """
    Plot 1: Main comparison - Accuracy vs M with confidence intervals
    Modern, clean style with gradient fill between methods.
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    M_values = ALL_M

    # Plot with gradient fill
    ours_acc = np.array(ALL_ACC['variance_centroid']) * 100
    random_acc = np.array(ALL_ACC['random']) * 100

    # Fill between our method and random
    ax.fill_between(M_values, ours_acc, random_acc,
                    alpha=0.15, color=COLORS['ours'],
                    label='_nolegend_')

    # Plot lines
    ax.plot(M_values, ours_acc,
            color=COLORS['ours'], marker='o', markersize=12,
            linewidth=3, label='Variance-Weighted Centroid (Ours)',
            markeredgecolor='white', markeredgewidth=2)

    ax.plot(M_values, np.array(ALL_ACC['cluster_boundary']) * 100,
            color=COLORS['boundary'], marker='s', markersize=11,
            linewidth=3, label='Cluster Boundary (Ours)',
            markeredgecolor='white', markeredgewidth=2)

    ax.plot(M_values, random_acc,
            color=COLORS['random'], marker='^', markersize=11,
            linewidth=3, label='Random Selection',
            markeredgecolor='white', markeredgewidth=2)

    # Styling
    ax.set_xscale('log')
    ax.set_xlabel('Number of Prototypes (M)', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title('Prototype Selection Performance on MNIST', fontsize=18, pad=20)

    ax.set_xticks(M_values)
    ax.set_xticklabels([f'{m:,}' for m in M_values])
    ax.set_ylim([70, 98])

    # Add compression ratio as secondary x-axis info
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(M_values)
    ax2.set_xticklabels([f'{60000//m}x' for m in M_values])
    ax2.set_xlabel('Compression Ratio', fontsize=12, color=COLORS['gray'])
    ax2.tick_params(colors=COLORS['gray'])

    # Legend
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True,
                       shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(COLORS['light_gray'])

    # Add annotations for key improvements
    ax.annotate('+13.8%', xy=(100, 86.5), xytext=(70, 90),
                fontsize=14, fontweight='bold', color=COLORS['ours'],
                arrowprops=dict(arrowstyle='->', color=COLORS['ours'], lw=2))

    ax.annotate('+3.9%', xy=(1000, 92.5), xytext=(700, 95),
                fontsize=12, fontweight='bold', color=COLORS['ours'],
                arrowprops=dict(arrowstyle='->', color=COLORS['ours'], lw=1.5))

    plt.tight_layout()
    plt.savefig('../figures/main_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('../figures/main_comparison.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: main_comparison.png/pdf")
    plt.close()


def plot_improvement_bar():
    """
    Plot 2: Improvement over random - horizontal bar chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    M_values = [100, 500, 1000, 5000, 10000]
    improvements = []
    for M, ours, rand in zip(ALL_M, ALL_ACC['variance_centroid'], ALL_ACC['random']):
        improvements.append((ours - rand) * 100)

    # Reverse for display (small M at top)
    M_labels = [f'M = {m:,}' for m in M_values]
    imp_values = improvements[::-1]

    # Gradient colors based on value
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp_values)))

    bars = ax.barh(M_labels, imp_values, color=colors, height=0.6, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, imp_values):
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                f'+{val:.1f}%', va='center', fontsize=13, fontweight='bold',
                color=COLORS['ours'])

    ax.set_xlabel('Accuracy Improvement over Random (%)', fontweight='bold')
    ax.set_title('Our Method vs Random Selection', fontsize=16, pad=15)
    ax.set_xlim([0, 16])
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/improvement_bar.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('../figures/improvement_bar.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: improvement_bar.png/pdf")
    plt.close()


def plot_per_class_comparison():
    """
    Plot 3: Per-class accuracy comparison - grouped bar chart
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    classes = list(range(10))
    ours_acc = [RESULTS['per_class_accuracy'][1000]['variance_centroid'][c] * 100 for c in classes]
    rand_acc = [RESULTS['per_class_accuracy'][1000]['random'][c] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, ours_acc, width, label='Ours (Variance-Weighted)',
                   color=COLORS['ours'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, rand_acc, width, label='Random',
                   color=COLORS['random'], alpha=0.7, edgecolor='white', linewidth=1.5)

    # Highlight improvements
    for i, (o, r) in enumerate(zip(ours_acc, rand_acc)):
        if o - r > 5:  # Significant improvement
            ax.annotate(f'+{o-r:.1f}%', xy=(i, max(o, r) + 1),
                        ha='center', fontsize=9, fontweight='bold',
                        color=COLORS['ours'])

    ax.set_xlabel('Digit Class', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Per-Class Accuracy (M = 1,000)', fontsize=16, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim([75, 102])
    ax.legend(loc='lower right', frameon=True)

    # Reference line
    ax.axhline(y=90, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.5)
    ax.text(9.5, 90.5, '90%', va='bottom', ha='right', color=COLORS['gray'])

    plt.tight_layout()
    plt.savefig('../figures/per_class_accuracy.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('../figures/per_class_accuracy.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: per_class_accuracy.png/pdf")
    plt.close()


def plot_prototype_distribution():
    """
    Plot 4: Prototype distribution - radar/polar chart style comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    classes = list(range(10))
    ours_dist = [RESULTS['prototype_distribution'][1000]['variance_centroid'][c] for c in classes]
    rand_dist = [RESULTS['prototype_distribution'][1000]['random'][c] for c in classes]

    # Left: Our method
    ax1 = axes[0]
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, 10))
    bars = ax1.bar(classes, ours_dist, color=colors, edgecolor='white', linewidth=2)
    ax1.axhline(y=100, color=COLORS['accent'], linestyle='--', linewidth=2, label='Uniform (100)')
    ax1.set_xlabel('Digit Class', fontweight='bold')
    ax1.set_ylabel('Number of Prototypes', fontweight='bold')
    ax1.set_title('Our Method: Variance-Weighted Allocation', fontsize=14, pad=10)
    ax1.set_ylim([0, 150])
    ax1.legend()

    # Add annotations for min/max
    min_idx = np.argmin(ours_dist)
    max_idx = np.argmax(ours_dist)
    ax1.annotate(f'Min: {ours_dist[min_idx]}', xy=(min_idx, ours_dist[min_idx]),
                 xytext=(min_idx, ours_dist[min_idx] + 15),
                 ha='center', fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate(f'Max: {ours_dist[max_idx]}', xy=(max_idx, ours_dist[max_idx]),
                 xytext=(max_idx, ours_dist[max_idx] + 15),
                 ha='center', fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    # Right: Random
    ax2 = axes[1]
    colors = plt.cm.Reds(np.linspace(0.3, 0.7, 10))
    bars = ax2.bar(classes, rand_dist, color=colors, edgecolor='white', linewidth=2)
    ax2.axhline(y=100, color=COLORS['accent'], linestyle='--', linewidth=2, label='Uniform (100)')
    ax2.set_xlabel('Digit Class', fontweight='bold')
    ax2.set_ylabel('Number of Prototypes', fontweight='bold')
    ax2.set_title('Random: Approximately Uniform', fontsize=14, pad=10)
    ax2.set_ylim([0, 150])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('../figures/prototype_distribution.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('../figures/prototype_distribution.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: prototype_distribution.png/pdf")
    plt.close()


def plot_tradeoff_curve():
    """
    Plot 5: Accuracy vs Compression tradeoff curve
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Compression ratios
    compression = [60000 // M for M in ALL_M]

    ours_acc = np.array(ALL_ACC['variance_centroid']) * 100
    rand_acc = np.array(ALL_ACC['random']) * 100

    # Plot curves
    ax.plot(compression, ours_acc, color=COLORS['ours'], marker='o', markersize=12,
            linewidth=3, label='Our Method', markeredgecolor='white', markeredgewidth=2)
    ax.plot(compression, rand_acc, color=COLORS['random'], marker='^', markersize=11,
            linewidth=3, label='Random', markeredgecolor='white', markeredgewidth=2)

    # Fill area
    ax.fill_between(compression, ours_acc, rand_acc,
                    alpha=0.2, color=COLORS['ours'])

    # Reference line for full 1-NN
    ax.axhline(y=97, color=COLORS['gray'], linestyle=':', linewidth=2)
    ax.text(620, 97.3, 'Full 1-NN (~97%)', fontsize=11, color=COLORS['gray'])

    ax.set_xlabel('Compression Ratio', fontweight='bold', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=14)
    ax.set_title('Accuracy-Compression Trade-off', fontsize=18, pad=20)

    ax.set_xscale('log')
    ax.set_xticks(compression)
    ax.set_xticklabels([f'{c}x' for c in compression])
    ax.set_ylim([70, 100])

    ax.legend(loc='lower left', frameon=True, fontsize=12)

    # Annotate key points
    ax.annotate('60x compression\n92.5% accuracy', xy=(60, 92.5),
                xytext=(100, 85), fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['ours'], lw=2),
                color=COLORS['ours'], fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/tradeoff_curve.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('../figures/tradeoff_curve.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: tradeoff_curve.png/pdf")
    plt.close()


def plot_summary_dashboard():
    """
    Plot 6: Summary dashboard with multiple subplots
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Main accuracy comparison (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    M_values = ALL_M
    ax1.plot(M_values, np.array(ALL_ACC['variance_centroid']) * 100,
             color=COLORS['ours'], marker='o', markersize=10, linewidth=2.5,
             label='Our Method', markeredgecolor='white', markeredgewidth=1.5)
    ax1.plot(M_values, np.array(ALL_ACC['random']) * 100,
             color=COLORS['random'], marker='^', markersize=9, linewidth=2.5,
             label='Random', markeredgecolor='white', markeredgewidth=1.5)
    ax1.fill_between(M_values,
                     np.array(ALL_ACC['variance_centroid']) * 100,
                     np.array(ALL_ACC['random']) * 100,
                     alpha=0.15, color=COLORS['ours'])
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Prototypes (M)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('A. Accuracy vs Prototypes', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xticks(M_values)
    ax1.set_xticklabels([str(m) for m in M_values])
    ax1.set_ylim([70, 98])

    # Plot 2: Improvement bars (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    improvements = [(o - r) * 100 for o, r in zip(ALL_ACC['variance_centroid'], ALL_ACC['random'])]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(improvements)))
    bars = ax2.barh([f'{m}' for m in ALL_M], improvements, color=colors, height=0.6)
    ax2.set_xlabel('Improvement (%)')
    ax2.set_title('B. Improvement over Random', fontweight='bold')
    for bar, val in zip(bars, improvements):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                 f'+{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax2.set_xlim([0, 16])

    # Plot 3: Per-class accuracy (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    classes = list(range(10))
    ours_acc = [RESULTS['per_class_accuracy'][1000]['variance_centroid'][c] * 100 for c in classes]
    rand_acc = [RESULTS['per_class_accuracy'][1000]['random'][c] * 100 for c in classes]
    x = np.arange(len(classes))
    width = 0.35
    ax3.bar(x - width/2, ours_acc, width, label='Ours', color=COLORS['ours'])
    ax3.bar(x + width/2, rand_acc, width, label='Random', color=COLORS['random'], alpha=0.7)
    ax3.set_xlabel('Digit')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('C. Per-Class (M=1000)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes)
    ax3.set_ylim([75, 102])
    ax3.legend(fontsize=9)

    # Plot 4: Prototype distribution (bottom-middle)
    ax4 = fig.add_subplot(gs[1, 1])
    ours_dist = [RESULTS['prototype_distribution'][1000]['variance_centroid'][c] for c in classes]
    ax4.bar(classes, ours_dist, color=plt.cm.Blues(np.linspace(0.3, 0.8, 10)), edgecolor='white')
    ax4.axhline(y=100, color=COLORS['accent'], linestyle='--', linewidth=2)
    ax4.set_xlabel('Digit')
    ax4.set_ylabel('Prototypes')
    ax4.set_title('D. Prototype Allocation', fontweight='bold')
    ax4.set_ylim([0, 140])

    # Plot 5: Key metrics (bottom-right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    # Add text summary
    summary_text = """
    Key Results (M = 1,000)

    Our Method:    92.47% accuracy
    Random:        88.56% accuracy
    Improvement:   +3.91%

    Compression:   60x
    (60,000 → 1,000 prototypes)

    Selection Time: ~15 seconds
    """
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray', alpha=0.8))
    ax5.set_title('E. Summary', fontweight='bold')

    plt.suptitle('Prototype Selection for 1-NN Classification on MNIST',
                 fontsize=20, fontweight='bold', y=0.98)

    plt.savefig('../figures/summary_dashboard.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('../figures/summary_dashboard.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved: summary_dashboard.png/pdf")
    plt.close()


def main():
    """Generate all plots."""
    import os
    os.makedirs('../figures', exist_ok=True)

    print("=" * 50)
    print("Generating Visualizations")
    print("=" * 50)

    plot_main_comparison()
    plot_improvement_bar()
    plot_per_class_comparison()
    plot_prototype_distribution()
    plot_tradeoff_curve()
    plot_summary_dashboard()

    print("=" * 50)
    print("All figures saved to ../figures/")
    print("=" * 50)


if __name__ == '__main__':
    main()
