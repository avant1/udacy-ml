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
clusters = model.predict(X).reshape(-1, 1)
cluster_numbers_with_initial_data = np.append(dataset, clusters, axis=1)

# agrrh, use get_named_colors_mapping instead
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

for row in cluster_numbers_with_initial_data:
    cluster_number = row[-1]
    color = colors[cluster_number]
    year_income = row[-3]
    spending_score = row[-2]
    plt.scatter(year_income, spending_score, color=color)

plt.show()
