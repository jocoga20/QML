import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import transforms, datasets
from deltakmeans import DeltaKMeans
from kmeans import KMeans
import time

def get_mnist():
    m = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
    return m.data.numpy().reshape(-1, 784), m.targets.numpy()

X, y = get_mnist()

scaler = StandardScaler()
X = scaler.fit_transform(X)
d = 40
pca = PCA(d)

pca.fit(X)
X = pca.transform(X)

print(X.shape)
print(pca.explained_variance_ratio_.sum())

k = 320
n = 60_000

km = DeltaKMeans(delta=.5)
c = km.kmeanspp_init(vectors=X, k=k)
t0 = time.time()

labels, centroids = km.run(vectors=X, centroids=c, labels=np.zeros(n), max_it=100, threshold=1e-4)

def get_centroids_majority_class(labels, y, k):
    return [np.bincount(y[labels == i]).argmax() for i in range(k)]

majority_class_by_centroid = get_centroids_majority_class(labels, y, k)

accuracy = 0.

for i in range(n):
    centroid_index = labels[i]
    majority_class = majority_class_by_centroid[centroid_index]
    if majority_class == y[i]:
        accuracy += 1

accuracy /= 60_000
print(accuracy)
t1 = time.time()
print(f't: {t1 - t0}')