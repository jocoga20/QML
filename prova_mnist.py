import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import transforms, datasets
from deltakmeans import DeltaKMeans
from kmeans import KMeans

def get_mnist(train):
    m = datasets.MNIST('./data', train=train, transform=transforms.ToTensor(), download=True)
    return m.data.numpy().reshape(-1, 784), m.targets.numpy()
d = 40
"""
X_tr, y_tr = get_mnist(True)
X_ts, y_ts = get_mnist(False)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_ts = scaler.transform(X_ts)
"""
"""
X_tr = np.load('x.train.std.npy')
X_ts = np.load('x.test.std.npy')

pca = PCA(d)
X_tr = pca.fit_transform(X_tr)
X_ts = pca.transform(X_ts)
"""

X_tr = np.load('x.train.40.npy')
X_ts = np.load('x.test.40.npy')
y_tr = np.load('y.train.npy')
y_ts = np.load('y.test.npy')

k = int(d / 2)
km = KMeans()
km = DeltaKMeans(delta=.5)
c = km.kmeanspp_init(vectors=X_tr, k=k)

labels, centroids = km.run(vectors=X_tr, centroids=c, labels=np.zeros(shape=X_tr.shape[0]), max_it=50, threshold=1e-4)

def get_centroids_majority_class(labels, y, k):
    return [np.bincount(y[labels == i]).argmax() for i in range(k)]

def nearest_centroid_index(vector, centroids):
    return np.linalg.norm(centroids - vector, axis=1).argmin()

majority_class_by_centroid = get_centroids_majority_class(labels, y_tr, k)

def accuracy(X, y, centroids):
    preds = np.array([majority_class_by_centroid[nearest_centroid_index(x, centroids)] for x in X])
    return (preds == y).sum() / X.shape[0]

print(f'TRAIN: {accuracy(X_tr, y_tr, centroids)}')
print(f'TEST: {accuracy(X_ts, y_ts, centroids)}')
