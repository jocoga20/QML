import numpy as np

def vector_potential(vector, centroid):
    m = vector - centroid
    return np.dot(m.T, m)

def potential(vectors, centroids, labels):
    return sum([vector_potential(vectors[i], centroids[labels[i]]) for i in range(vectors.shape[0])])