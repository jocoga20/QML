import numpy as np
from time import sleep
from plot import plot2d
from random import randint

def labeling(vectors, centroids):
    return np.array([np.linalg.norm(centroids - v, axis=1).argmin() for v in vectors])

def compute_centroids(vectors, labels, k):
    return np.array([vectors[labels==label].mean(axis=0) for label in range(k)])

<<<<<<< HEAD
def kmeanspp_init_centroids(vectors, k):
    i = randint(0, vectors.shape[0]-1)
    c0 = vectors[i]
    

def kmeans(vectors, centroids, labels, iterations=1, threshold=float('-inf')):
=======
def kmeans(vectors, centroids, iterations=1, threshold = float('-inf')):
>>>>>>> e6d57b54da836f310a43063ab6507a05fea70f31
    """
    :param vectors: n x d n = instances, d = feature size
    :param centroids: k x d, k = num of different labels (given)
    :param labels: n-dim array = each element is the label assigned to the corresponding vector
    :param iterations: num of iterations required
    :param threshold: when average movement of centroids is no longer bigger then threshold, algorithm stops. Default is -inf (no stop).

    :returns: labels, centroids
    """
    k = centroids.shape[0]
<<<<<<< HEAD
    threshold *= k
    
    for _ in range(iterations):
        labels = labeling(vectors, centroids)
        new_centroids = compute_centroids(vectors, labels, k)
        if np.linalg.norm(new_centroids - centroids, axis=1).sum() <= threshold:
            break
        centroids = new_centroids

    
    return labels, new_centroids
=======
    labels = np.zeros(n)

    # to use .sum() instead of .mean() we modify the threshold condition
    threshold *= k

    for i in range(iterations):
        labels = labeling(vectors, centroids)
        new_centroids = compute_centroids(vectors, labels, k)
        mean_move = new_centroids - centroids
        mean_move = np.linalg.norm(mean_move, axis=1)
        mean_move = np.sum(mean_move)
        centroids = new_centroids
        if mean_move <= threshold:
            break
    
    return labels, centroids

x = np.array([[0, 0, 0], [1, 1, 1], [1,2,3], [3,3,3]])
y = np.array([[1, 0, 0], [1, -2, 1], [0,2,4], [3,3,3]])

print(x - y)

print(np.linalg.norm(x - y, axis=1))
>>>>>>> e6d57b54da836f310a43063ab6507a05fea70f31
