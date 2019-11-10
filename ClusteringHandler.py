import dippykit
import sklearn.cluster as sk_cluster
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import numpy as np


class ClusteringHandler():

    def __init__(self, data, labels=None, min_clusters=2, max_clusters=10):
        self.data = np.asarray(data)
        self.data.sort()
        self.labels = labels
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.num_clusters = None
        self.cluster_centers = None

    def cluster_data(self, method='k_means'):
        if method == "k_means":
            self.k_means_cluster()

    def k_means_cluster(self):

        best_score = -1
        best_result = None
        current_result = None
        current_score = None
        for i in range(self.min_clusters, np.min([self.max_clusters+1, self.data.shape[0]])):
            kmeans_model = sk_cluster.KMeans(n_clusters=i)
            current_result = kmeans_model.fit(self.data)
            current_labels = current_result.labels_
            current_score = self.score_clustering(current_labels)

            if current_score > best_score:
                best_score = current_score
                best_result = current_result
                self.num_clusters = i
                self.labels = best_result.labels_
                self.cluster_centers = best_result.cluster_centers_

    def score_clustering(self, labels):
        return sk_metrics.silhouette_score(self.data, labels)

    def plot_clusters(self):
        pass
        #TODO: possibly unnecessary plotting of the clustering.


if __name__ == "__main__":
    data = np.asarray([
        [1, 2, 3],
        [3, 2, 1],
        [3, 3, 3],
        [1, 1, 1]
    ])
    ch = ClusteringHandler(data)
    ch.cluster_data()
    print(ch.labels)
