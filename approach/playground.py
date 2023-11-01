import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Example distance matrix
dist_matrix = np.array([
    [0., 4.03536882, 2.48645379, 2.82112646, 3.96958974, 3.15893184, 3.17685658, 1.7205738],
    [4.03536882, 0., 3.89288972, 6.84135136, 7.15464916, 7.13626081, 2.43648418, 2.75199777],
    [2.48645379, 3.89288972, 0., 3.1839766, 3.72308007, 3.5961519, 1.47951198, 0.94499946],
    [2.82112646, 6.84135136, 3.1839766, 0., 1.21608752, 0.59725393, 4.17393569, 3.16290091],
    [3.96958974, 7.15464916, 3.72308007, 1.21608752, 0., 0.76739677, 4.68575211, 3.95765163],
    [3.15893184, 7.13626081, 3.5961519, 0.59725393, 0.76739677, 0., 4.55622635, 3.51612036],
    [3.17685658, 2.43648418, 1.47951198, 4.17393569, 4.68575211, 4.55622635, 0., 1.55638223],
    [1.7205738, 2.75199777, 0.94499946, 3.16290091, 3.95765163, 3.51612036, 1.55638223, 0.]
])

# Set your desired threshold
threshold = 2

# Perform hierarchical clustering
linkage_matrix = linkage(dist_matrix, method='single', optimal_ordering=True, metric='euclidean')
print(linkage_matrix)

# Get cluster labels
print(linkage_matrix) 
print(threshold)
labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
print(labels)

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, color_threshold=threshold)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.ylim([0, 10])  # Adjust the y-axis limits
plt.show()

# Output the cluster labels
print("Cluster labels:", labels)
