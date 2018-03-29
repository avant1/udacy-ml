import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv').as_matrix()
X = dataset[:, -2:].astype(int)

# elbow method
# wcss = []
# inspected_cluster_numbers = range(1, 11)
# for i in inspected_cluster_numbers:
#     model = KMeans(n_clusters=i, random_state=0)
#     model.fit(X)
#     score = model.inertia_
#     wcss.append(score)
#
# plt.plot(inspected_cluster_numbers, wcss)
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.title('Elbow method')
# plt.show()

optimal_number_of_clusters = 5
model = KMeans(n_clusters=optimal_number_of_clusters, random_state=0)
model.fit(X)
clusters = model.predict(X)

# agrrh, use get_named_colors_mapping instead
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

for cluster_number in range(0, optimal_number_of_clusters):
    color = colors[cluster_number]
    plt.scatter(
        X[clusters == cluster_number, 0],
        X[clusters == cluster_number, 1],
        s=50,
        color=color,
        label='Cluster {}'.format(cluster_number)
    )

plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=150, color='yellow', label='Centroids')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score')
plt.legend()
plt.show()
