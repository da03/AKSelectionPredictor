import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Example precision and recall data
precisions = np.random.rand(16)  # Replace with your actual precision data
recalls = np.random.rand(16)     # Replace with your actual recall data
ks = list(range(1, 32, 2))          # k values from 1 to 20

# Create a color map and normalize
norm = plt.Normalize(min(ks), max(ks))  # Normalization for the color map
colors = cm.viridis(norm(ks))  # Get color map colors based on normalized k values

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))
sc = ax.scatter(recalls, precisions, color=colors, s=50)  # Plot scatter with color

# Adding annotations for each point
for i, k in enumerate(ks):
    ax.text(recalls[i], precisions[i], f'{k}', fontsize=9, ha='right', va='center')

# Adding labels and title
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision vs Recall for Different k in k-NN')

# Create a ScalarMappable and Colorbar
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])  # You need to set_array([]) to avoid warnings in some Matplotlib versions
cbar = plt.colorbar(sm, ax=ax, aspect=10)  # Link the colorbar to the ax
cbar.set_label('Value of k')

plt.savefig('test_plot.png')
