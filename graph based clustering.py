import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.datasets import make_blobs
from mst_clustering import MSTClustering

# Suppress specific warnings
warnings.filterwarnings("ignore", message="elementwise")

# Function to plot the MST clustering results
def plot_mst(model, cmap='rainbow'):
    """
    Plot the MST clustering results: Full Minimum Spanning Tree and Trimmed Minimum Spanning Tree.
    """
    X = model.X_fit_  # Get the fitted data
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    for axi, full_graph, colors in zip(ax, [True, False], ['lightblue', model.labels_]):
        segments = model.get_graph_segments(full_graph=full_graph)  # Get graph segments
        axi.plot(segments[0], segments[1], '-k', zorder=1, lw=1)  # Plot the edges of the MST
        
        # Plot the data points with colors
        if isinstance(colors, np.ndarray):
            axi.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, zorder=2, edgecolor='k')
        else:
            axi.scatter(X[:, 0], X[:, 1], c='lightblue', zorder=2, edgecolor='k')
        
        axi.axis('tight')
    ax[0].set_title('Full Minimum Spanning Tree', size=16)
    ax[1].set_title('Trimmed Minimum Spanning Tree', size=16)
    plt.tight_layout()
    plt.show()

# Create synthetic data with blobs
X, y = make_blobs(200, centers=6, random_state=42)

# Visualize the synthetic data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue')
plt.title('Synthetic Data')
plt.show()

# Initialize the MST Clustering model
model = MSTClustering(cutoff_scale=2, approximate=False)

# Fit the model and predict clusters
labels = model.fit_predict(X)

# Visualize the clustered data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title('Clustered Data')
plt.show()

# Plot the MST results: Full and Trimmed Trees
plot_mst(model)