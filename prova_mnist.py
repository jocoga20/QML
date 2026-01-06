import numpy as np
#from sklearn.decomposition import PCA
from torchvision import transforms, datasets

from kmeans import KMeans

mnist = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
X = mnist.data.numpy().reshape(-1, 784)
y = mnist.targets.numpy()

k = 10
n = 60_000
d = 784

km = KMeans()
labels, centroids = km.run(vectors=X, centroids=km.kmeanspp_init(vectors=X, k=k), labels=np.zeros(n), iterations=10, threshold=1e-4)

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