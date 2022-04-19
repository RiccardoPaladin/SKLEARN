import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wine = datasets.load_wine(as_frame= True)
wine_data = wine.data


def best_number_of_clusters(wine_data):
    '''
    Function that look for the best number of clusters in the KMeans unsupervised machine
    Input: wine data
    Output: plot with relationship between SSE and N° of clusters, hilighted the best number
    '''
    sum_squered_error = []
    n_cluster = []
    for k in range(1, 8):
        kmeans = KMeans(n_clusters=k)
        fit = kmeans.fit(wine_data)
        sum_squered_error.append(fit.inertia_)
        n_cluster.append(k)
    plt.plot(n_cluster, sum_squered_error, 'b', marker='o')
    plt.plot(4, sum_squered_error[3], marker="o", markersize=10, markerfacecolor="red")
    plt.title('The Elbow Method showing the optimal k', fontsize = 10)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Error')
    plt.suptitle('Based on the plot below, the best number of clusters to reduce the error is 4', fontsize = 10)
    plt.show()

if __name__ == '__main__':
    best_number_of_clusters(wine_data)