from kmeans import kmeans, compute_centroids
import numpy as np
from plot import plot2d
from utils import potential

#print(kmeans(np.ones((100, 3)), np.ones((5, 3)), 1_000))

d = 2
n = 1000
k = 4

def normal(mean, num):
    return np.random.multivariate_normal(mean=mean, cov=np.identity(d) * 0.1, size=num)

vectors = np.concat([
    normal(np.zeros(d), int(n/3)),
    normal(np.ones(d) * 3, int(n/3)),
    normal(np.ones(d) * -3, n - 2 * int(n/3))
], axis=0)

labels = np.array([i % k for i in range(n)])
centroids = compute_centroids(vectors, labels, k)

p = potential(vectors, centroids, labels)
plot2d(vectors, labels, centroids, k, title=f'it=0, p={p}')

for it in range(1, 100):
    labels, centroids = kmeans(vectors, centroids, labels)
    p = potential(vectors, centroids, labels)
    plot2d(vectors, labels, centroids, k, title=f'it={it}, p={p}')
    
