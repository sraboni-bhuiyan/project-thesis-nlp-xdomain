import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load in-domain metrics
metrics_summary = pd.read_csv('./metrics_summary.csv')

# Calculate averages per model
model_stats = metrics_summary.groupby('model').agg({
    'training_time_s': 'mean',
    'macro_f1': 'mean',
    'model_params': 'first'
}).reset_index()

# Convert to minutes
model_stats['training_time_min'] = model_stats['training_time_s'] / 60

# Prepare data
models_display = {
    'bert': 'BERT',
    'roberta': 'RoBERTa',
    'electra': 'ELECTRA',
    'distilbert': 'DistilBERT',
    'albert': 'ALBERT',
    'xlnet': 'XLNet'
}

colors = {
    'bert': '#1f77b4',
    'roberta': '#ff7f0e',
    'electra': '#2ca02c',
    'distilbert': '#d62728',
    'albert': '#9467bd',
    'xlnet': '#8c564b'
}

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each model
for _, row in model_stats.iterrows():
    model_key = row['model']
    model_name = models_display[model_key]
    
    # Size based on parameters (scaled for visibility)
    size = (row['model_params'] / 1e6) * 3  # Scale for better visualization
    
    ax.scatter(row['training_time_min'], row['macro_f1'], 
               s=size, alpha=0.7, color=colors[model_key],
               edgecolors='black', linewidth=2, label=model_name)
    
    # Add model name label
    ax.annotate(model_name, 
                xy=(row['training_time_min'], row['macro_f1']),
                xytext=(10, 5), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[model_key], 
                         alpha=0.3, edgecolor=colors[model_key]))

# Add quadrant lines
median_time = model_stats['training_time_min'].median()
median_f1 = model_stats['macro_f1'].median()

ax.axvline(median_time, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axhline(median_f1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

# Quadrant labels
ax.text(median_time - 0.15, model_stats['macro_f1'].max() - 0.002,
        'High Performance\nHigh Cost', ha='right', va='top',
        fontsize=9, style='italic', alpha=0.6, bbox=dict(boxstyle='round', 
        facecolor='wheat', alpha=0.3))
ax.text(median_time + 0.15, model_stats['macro_f1'].max() - 0.0065,
        'High Performance\nLow Cost', ha='left', va='top',
        fontsize=9, style='italic', alpha=0.6, bbox=dict(boxstyle='round',
        facecolor='lightgreen', alpha=0.3))

# Styling
ax.set_xlabel('Training Time per Dataset (minutes)', fontsize=13, fontweight='bold')
ax.set_ylabel('Average In-Domain Macro-F1 Score', fontsize=13, fontweight='bold')
ax.set_title('Efficiency-Performance Trade-off of Transformer Models\n(Bubble size represents model parameters)',
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(1.5, 7.5)
ax.set_ylim(0.685, 0.750)

# Legend
legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')

plt.tight_layout()
plt.savefig('visualization_3_efficiency_tradeoff.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization 3 saved as 'visualization_3_efficiency_tradeoff.png'")
