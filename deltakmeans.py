import numpy as np
from random import choice
import kmeans.compute_centroids

def distance(x, y):
    return np.linalg.norm(x-y)

def assign_label(vector, centroids, delta):
    k = centroids.shape[0]
    d_min = float('inf')
    filtered_centroids = []
    distances = []

    for i in range(k):
        c = centroids[i]
        d = distance(vector, c)
        distances.append(d)
        if d > d_min + delta:
            continue
        if d < d_min:
            d_min = d
        filtered_centroids.append(i)
    
    filtered_centroids = [i for i in filtered_centroids if distances[i] <= d_min + delta]
    return choice(filtered_centroids)

def compute_centroids(vectors, labels, k, delta):
    d = vectors.shape[1]
    return kmeans.compute_centroids(vectors, labels, k) + delta / (2 * d)

def assign_labels(vectors, centroids, delta):
    return np.array([assign_label(v, centroids, delta) for v in vectors])

def deltakmeans(vectors, centroids, labels, delta, iterations=1, threshold=float('-inf')):
    k = centroids.shape[0]
    for _ in range(iterations):
        labels = assign_labels(vectors, centroids, delta)
        compute_centroids(vectors, labels, k, delta)
        

vs = np.array([[0,0], [1,1], [2,2]])
cs = np.array([[2,0], [2,1]])

l = assign_label(vs[0], cs, delta=1e-10)
print(l)