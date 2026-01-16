import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import transforms, datasets
from deltakmeans import DeltaKMeans

def get_mnist(train):
    m = datasets.MNIST('./data', train=train, transform=transforms.ToTensor(), download=True)
    return m.data.numpy().reshape(-1, 784), m.targets.numpy()

X_tr, y_tr = get_mnist(True)
X_ts, y_ts = get_mnist(False)
"""
print(X_tr.shape, np.linalg.matrix_rank(X_tr))
print(X_ts.shape, np.linalg.matrix_rank(X_ts))
import sys
sys.exit(0)
"""
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_ts = scaler.transform(X_ts)

d = 712
pca = PCA(d)
X_tr = pca.fit_transform(X_tr)
X_ts = pca.transform(X_ts)

k = 100

km = DeltaKMeans(delta=.5)
c = km.kmeanspp_init(vectors=X_tr, k=k)

labels, centroids = km.run(vectors=X_tr, centroids=c, labels=np.zeros(shape=X_tr.shape[0]), max_it=20, threshold=1e-10)

def get_centroids_majority_class(labels, y, k):
    return [np.bincount(y[labels == i]).argmax() for i in range(k)]

def nearest_centroid_index(vector, centroids):
    return np.linalg.norm(centroids - vector, axis=1).argmin()

def accuracy(X, y, centroids):
    preds = []
    for x in X:
        i = nearest_centroid_index(x, centroids)
        preds.append(majority_class_by_centroid[i])
    preds = np.array(preds)
    return (preds == y).sum() / X.shape[0]

majority_class_by_centroid = get_centroids_majority_class(labels, y_tr, k)

print(accuracy(X_tr, y_tr, centroids))
print(accuracy(X_ts, y_ts, centroids))
