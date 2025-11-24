import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load cross-domain results
cross_domain = pd.read_csv('./cross_domain_results.csv')

# Create pivot table for heatmap (average across all models)
heatmap_data = cross_domain.pivot_table(
    values='macro_f1',
    index='source_dataset',
    columns='target_dataset',
    aggfunc='mean'
)

# Reorder for better visualization
dataset_order = ['sentiment140', 'reddit', 'amazon_combined', 'imdb']
dataset_labels = ['Sentiment140\n(Social Media)', 'Reddit\n(Social Media)', 
                  'Amazon\n(Product Review)', 'IMDb\n(Product Review)']

heatmap_data = heatmap_data.reindex(index=dataset_order, columns=dataset_order)

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap
mask = np.eye(len(dataset_order), dtype=bool)  # Mask diagonal (in-domain)
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0, vmax=0.8, center=0.4,
            cbar_kws={'label': 'Average Macro-F1 Score'},
            linewidths=2, linecolor='white',
            mask=mask, ax=ax,
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})

# Set labels
ax.set_xlabel('Target Dataset (Test)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Source Dataset (Train)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Cross-Domain Transfer Performance Heatmap\n(Average Macro-F1 Across All Models)', 
             fontsize=15, fontweight='bold', pad=20)

# Set tick labels
ax.set_xticklabels(dataset_labels, fontsize=10, rotation=0)
ax.set_yticklabels(dataset_labels, fontsize=10, rotation=0)

# Add annotations for domain types
ax.text(-0.5, 1, 'Social Media', fontsize=11, fontweight='bold', 
        color='#E63946', rotation=90, va='center')
ax.text(-0.5, 3, 'Product Reviews', fontsize=11, fontweight='bold',
        color='#2E86AB', rotation=90, va='center')
ax.text(1, -0.5, 'Social Media', fontsize=11, fontweight='bold',
        color='#E63946', ha='center')
ax.text(3, -0.5, 'Product Reviews', fontsize=11, fontweight='bold',
        color='#2E86AB', ha='center')

plt.tight_layout()
plt.savefig('visualization_2_transfer_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization 2 saved as 'visualization_2_transfer_heatmap.png'")
