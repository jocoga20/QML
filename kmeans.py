import numpy as np
from time import sleep
from plot import plot2d

def labeling(vectors, centroids):
    return np.array([np.linalg.norm(centroids - v, axis=1).argmin() for v in vectors])

def compute_centroids(vectors, labels, k):
    return np.array([vectors[labels==label].sum(axis=0) for label in range(k)])

def kmeans(vectors, centroids, iterations=1):
    """
    :param vectors: n x d n = instances, d = feature size
    :param centroids: k x d, k = num of different labels (given)
    :param iterations: num of iterations required

    :returns: labels, centroids
    """
    n, d = vectors.shape
    k = centroids.shape[0]
    labels = np.zeros(n)
    
    for i in range(iterations):
        labels = labeling(vectors, centroids)
        centroids = compute_centroids(vectors, labels, k)
    
    return labels, centroids