import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

dataset = pd.read_csv('Mall_Customers.csv').as_matrix()
X = dataset[:, -2:].astype(int)

# # building dendrogram to choose optimal number of clusters
# dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
#
# plt.title('Dendrogram')
# plt.xlabel('Customer income, k$')
# plt.ylabel('Euclidian distances')
# plt.show()

model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
detected_clusters = model.fit_predict(X)
cluster_numbers = range(0, 5)

markers = ['s', 'p', 'o', 'd', '*']
colors = ['m', 'b', 'y', 'c', 'r', 'g']

for i in cluster_numbers:
    marker = markers[i]
    color = colors[i]
    plt.scatter(
        X[i == detected_clusters, 0],
        X[i == detected_clusters, 1],
        label='Cluster ' + str(i),
        marker=marker,
        s=50,
        color=color
    )

plt.title('Agglomerative Hierarchical Clustering')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score')
plt.legend()
plt.show()
