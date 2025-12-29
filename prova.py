from kmeans import kmeans, compute_centroids
import numpy as np
from plot import plot2d

#print(kmeans(np.ones((100, 3)), np.ones((5, 3)), 1_000))

d = 2
n = 1000
k = 4

vectors = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=n)
labels = np.array([i % k for i in range(n)])
centroids = compute_centroids(vectors, labels, k)
iterations = 2

plot2d(vectors, labels, k)

print(labels.shape, centroids.shape)
for k in range(1):
    labels, centroids = kmeans(vectors, centroids)
    print(labels)
    print(centroids)