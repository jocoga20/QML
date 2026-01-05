import numpy as np
from time import sleep
from plot import plot2d
from random import randint

def assign_labels(vectors, centroids):
    return np.array([np.linalg.norm(centroids - v, axis=1).argmin() for v in vectors])

def update_centroids(vectors, labels, k):
    return np.array([vectors[labels==label].mean(axis=0) for label in range(k)])

def min_distance(vector, centroids):
    return np.linalg.norm(vector - centroids, axis=1).min()

def nearest_centroid_distances(vectors, centroids):
    return np.array([min_distance(v, centroids) for v in vectors])

def kmeanspp_init_centroids(vectors, k):
    i = randint(0, vectors.shape[0]-1)
    c0 = vectors[i]
    

def kmeans(vectors, centroids, labels, iterations=1, threshold=float('-inf')):
    """
    :param vectors: n x d n = instances, d = feature size
    :param centroids: k x d, k = num of different labels (given)
    :param max_it: num of iterations required
    :param threshold: when average movement of centroids is no longer bigger then threshold, algorithm stops. Default is -inf (no stop).

    :returns: labels, centroids
    """
    k = centroids.shape[0]
    threshold *= k
    
    for _ in range(iterations):
        labels = labeling(vectors, centroids)
        new_centroids = compute_centroids(vectors, labels, k)
        if np.linalg.norm(new_centroids - centroids, axis=1).sum() <= threshold:
            break
        centroids = new_centroids

    return labels, new_centroids