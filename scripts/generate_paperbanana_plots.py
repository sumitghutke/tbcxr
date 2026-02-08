
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create directory if not exists
output_dir = "outputimg"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the style for Academic Publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2) # Optimized for IEEE/Springer papers

# 1. Data Distribution (Based on research paper data: 771 Normal, 202 TB)
labels = ['Normal', 'Tuberculosis']
counts = [771, 202]
colors = ['#2ec4b6', '#e71d36'] # Modern Teal and Coral

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
plt.title('Dataset Class Distribution', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Number of Images', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add counts on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, yval, ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'paperbanana_data_dist.png'), dpi=300)
plt.close()

# 2. Confusion Matrix (Based on research paper data: 97% accuracy)
# Actual: Normal (771), TB (202)
# Recall Normal: 0.99 -> Correct: 763, Wrong: 8
# Recall TB: 0.88 -> Correct: 178, Wrong: 24
cm = np.array([[763, 8], [24, 178]])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'TB'], 
            yticklabels=['Normal', 'TB'],
            annot_kws={"size": 16, "weight": "bold"})

plt.title('Confusion Matrix: TB-Ray Validation Set', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'paperbanana_confusion_matrix.png'), dpi=300)
plt.close()

print(f"Statistical plots generated successfully in {output_dir}")
