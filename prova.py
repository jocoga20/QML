from kmeans import assign_labels, kmeans, kmeanspp_init_centroids, update_centroids
import numpy as np
from plot import plot2d
from utils import potential

#print(kmeans(np.ones((100, 3)), np.ones((5, 3)), 1_000))

d = 20
n = 10000
k = 5

def normal(mean, num):
    return np.random.multivariate_normal(mean=mean, cov=np.identity(d) * 100, size=num)

vectors = np.concat([
    normal(np.zeros(d), int(n/3)),
    normal(np.ones(d) * 3, int(n/3)),
    normal(np.ones(d) * -3, n - 2 * int(n/3))
], axis=0)

centroids = kmeanspp_init_centroids(vectors, k)
labels, centroids = kmeans(vectors, centroids, max_it=20, threshold=1e-10)

"""
p = potential(vectors, centroids, labels)
plot2d(vectors, labels, centroids, k, title=f'it=0, p={p}')

for it in range(1, 100):
    labels, centroids = kmeans(vectors, centroids, labels)
    p = potential(vectors, centroids, labels)
    plot2d(vectors, labels, centroids, k, title=f'it={it}, p={p}')
    """
