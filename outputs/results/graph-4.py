import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load in-domain metrics
metrics_summary = pd.read_csv('./metrics_summary.csv')

# Prepare data
datasets_order = ['sentiment140', 'reddit', 'amazon_combined', 'imdb']
dataset_labels = ['Sentiment140\n(Social Media)', 'Reddit\n(Social Media)',
                  'Amazon\n(Product Review)', 'IMDb\n(Product Review)']

models_order = ['bert', 'roberta', 'electra', 'distilbert', 'albert', 'xlnet']
model_labels = ['BERT', 'RoBERTa', 'ELECTRA', 'DistilBERT', 'ALBERT', 'XLNet']

# Create pivot table
pivot_data = metrics_summary.pivot_table(
    values='macro_f1',
    index='model',
    columns='dataset'
)

# Reorder
pivot_data = pivot_data.reindex(index=models_order, columns=datasets_order)

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(datasets_order))
width = 0.13  # Width of bars

colors_models = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Plot bars for each model
for i, (model, color) in enumerate(zip(models_order, colors_models)):
    values = [pivot_data.loc[model, ds] for ds in datasets_order]
    offset = (i - len(models_order)/2 + 0.5) * width
    bars = ax.bar(x + offset, values, width, label=model_labels[i], 
                   color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0.5:  # Only label if value is reasonable
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7, rotation=90)

# Add domain separation
ax.axvline(1.5, color='black', linestyle='-', linewidth=2.5, alpha=0.7)
ax.text(0.75, 0.95, 'Social Media', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#E63946', alpha=0.3))
ax.text(2.25, 0.95, 'Product Reviews', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#2E86AB', alpha=0.3))

# Styling
ax.set_xlabel('Datasets', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Macro-F1 Score', fontsize=13, fontweight='bold')
ax.set_title('Dataset-Specific Model Performance Comparison\n(In-Domain Evaluation)',
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(dataset_labels, fontsize=11)
ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('visualization_4_dataset_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization 4 saved as 'visualization_4_dataset_model_comparison.png'")
