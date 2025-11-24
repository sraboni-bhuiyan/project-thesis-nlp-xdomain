import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
metrics_summary = pd.read_csv('./metrics_summary.csv')
cross_domain = pd.read_csv('./cross_domain_results.csv')

# Calculate in-domain average for each model
in_domain_avg = metrics_summary.groupby('model')['macro_f1'].mean()

# Calculate cross-domain average for each model
cross_domain_avg = cross_domain.groupby('model')['macro_f1'].mean()

# Prepare data for plotting
models = ['BERT', 'RoBERTa', 'ELECTRA', 'DistilBERT', 'ALBERT', 'XLNet']
model_keys = ['bert', 'roberta', 'electra', 'distilbert', 'albert', 'xlnet']

in_domain_values = [in_domain_avg[k] for k in model_keys]
cross_domain_values = [cross_domain_avg[k] for k in model_keys]
performance_gap = [in_domain_values[i] - cross_domain_values[i] for i in range(len(models))]

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(models))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, in_domain_values, width, label='In-Domain', color='#2E86AB', alpha=0.9)
bars2 = ax.bar(x + width/2, cross_domain_values, width, label='Cross-Domain', color='#E63946', alpha=0.9)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add gap annotations
for i, gap in enumerate(performance_gap):
    ax.annotate('', xy=(i, cross_domain_values[i]), xytext=(i, in_domain_values[i]),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(i + 0.25, (in_domain_values[i] + cross_domain_values[i])/2, 
            f'-{gap:.3f}', fontsize=8, color='darkred', fontweight='bold')

# Styling
ax.set_xlabel('Transformer Models', fontsize=13, fontweight='bold')
ax.set_ylabel('Macro-F1 Score', fontsize=13, fontweight='bold')
ax.set_title('In-Domain vs Cross-Domain Performance Gap by Model', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 0.85)

plt.tight_layout()
plt.savefig('visualization_1_performance_gap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization 1 saved as 'visualization_1_performance_gap.png'")
