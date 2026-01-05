import numpy as np

def vector_potential(vector, centroid):
    m = vector - centroid
    return np.dot(m.T, m)

def potential(vectors, centroids, labels):
    s = 0.
    for i in range(vectors.shape[0]):
        v = vectors[i]
        l = labels[i]
        c = centroids[l]
        m = v - c
        s += np.dot(m.T, m)
    return s