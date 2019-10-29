import dippykit
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np


class ClusteringHandler():

    def __init__(self, data, labels=None, min_clusters=1, max_clusters=10):
        self.data = data
        self.data.sort()
        self.labels = labels
        self.min_clusters = 1
        self.max_clusters = 10
        self.num_clusters = None

    def cluster_data(self, method='k_means'):
        if method == "knn":
            self.k_means_cluster()

    def k_means_cluster(self):

        best_score = -1
        best_result = None
        current_result = None
        current_score = None
        for i in range(self.min_clusters, self.max_clusters+1):
            kmeans_model = sk.cluster.KMeans(n_clusters=i)
            current_result = kmeans_model.fit(self.data)
            current_labels = current_result.labels_
            current_score = self.score_clustering(current_labels)

            if current_score > best_score:
                best_score = current_score
                best_result = current_result

        self.labels = best_result.labels_

    def score_clustering(self, labels):
        return sk.metrics.silhouette_score(self.data, labels)

    def plot_clusters(self):
        pass
        #TODO: possibly unnecessary plotting of the clustering.
