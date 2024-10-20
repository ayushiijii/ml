import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Load the dataset
# Use the correct path and file extension
df = pd.read_excel(r"C:\Users\Ayushi\Desktop\submissions\ML\Mall_Customers.xlsx")
print(df.head())
print("Dataset Shape: ", df.shape)

# Extract relevant features for clustering
x = df.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values
print(x.shape)

# Perform Nearest Neighbors
neighb = NearestNeighbors(n_neighbors=2)
nbrs = neighb.fit(x)
distances, indices = nbrs.kneighbors(x)

# Sort distances and get the second nearest distance (for DBSCAN epsilon)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]

# Plot distances
plt.rcParams["figure.figsize"] = (6, 3)
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('Second Nearest Distance')
plt.title('Nearest Neighbors Distances')
plt.show()

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=8, min_samples=4).fit(x)
labels = dbscan.labels_

# Plot the results
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="plasma")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering of Mall Customers")
plt.show()
