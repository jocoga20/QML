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

def accuracy(X, y, centroids):
    preds = np.array([majority_class_by_centroid[nearest_centroid_index(x, centroids)] for x in X])
    return (preds == y).sum() / X.shape[0]