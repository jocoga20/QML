import numpy as np

def potential(vectors, centroids, labels):
    s = 0.
    for i in range(vectors.shape[0]):
        v = vectors[i]
        l = labels[i]
        c = centroids[l]
        m = v - c
        s += np.dot(m.T, m)
    return s

def centroid_index_to_majority_class(i, labels, y_true):
    if all(labels != i):
        return -1
    total = (labels == i).sum()
    bins = np.bincount(y_true[labels == i])
    print(f'Purity of {i}: {bins.max() / total}')
    return bins.argmax().item()

def compute_centroids_classes(k, labels, y_true):
    return [centroid_index_to_majority_class(i, labels, y_true) for i in range(k)]

def nearest_centroid_index(vector, centroids):
    m = centroids - vector
    m **= 2
    m = m.sum(axis=1)
    m = m.argmin()
    return m

def vectors_to_class(vectors, centroids, centroid_index_to_class):
    return np.array([centroid_index_to_class[nearest_centroid_index(v, centroids)] for v in vectors])

def accuracy(y_pred, y_true):
    return (y_pred == y_true).sum() / y_pred.shape[0]