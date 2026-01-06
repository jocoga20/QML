import numpy as np

from plot import plot2d
from utils import potential

class KMeans:
    def get_sqrd_shortest_dist(self, vector, centroids):
        sqrd_dist = (centroids - vector) ** 2
        return sqrd_dist.sum(axis=1).min()
    
    def get_probs(self, vectors, centroids):
        sqrd_dists = np.array([self.get_sqrd_shortest_dist(v, centroids) for v in vectors]).astype('float64')
        return sqrd_dists / sqrd_dists.sum()

    def choose_new_index(self, vectors, centroids):
        n = vectors.shape[0]
        return np.random.choice(n, p=self.get_probs(vectors, centroids))

    def kmeanspp_init(self, vectors, k):
        n = vectors.shape[0]
        i = np.random.choice(n)
        centroids = [vectors[i]]
        
        for _ in range(k-1):
            i = self.choose_new_index(vectors, centroids)
            centroids.append(vectors[i])
        return np.array(centroids)

    def assign_labels(self, vectors, centroids):
        return np.array([np.linalg.norm(centroids - v, axis=1).argmin() for v in vectors])
    
    def update_centroids(self, vectors, labels, k):
        return np.array([vectors[labels==label].mean(axis=0) for label in range(k)])

    def run(self, vectors, centroids, labels, iterations=1, threshold=float('-inf')):
        """
        :param vectors: n x d n = instances, d = feature size
        :param centroids: k x d, k = num of different labels (given)
        :param max_it: num of iterations required
        :param threshold: when average movement of centroids is no longer bigger then threshold, algorithm stops. Default is -inf (no stop).

        :returns: labels, centroids
        """
        k = centroids.shape[0]
        threshold *= k
        
        for _ in range(iterations):
            labels = self.assign_labels(vectors, centroids)
            new_centroids = self.update_centroids(vectors, labels, k)
            if np.linalg.norm(new_centroids - centroids, axis=1).sum() <= threshold:
                break
            centroids = new_centroids

        return labels, new_centroids

n = 100
d = 2
k = 5

vectors = np.random.normal(size=(n, d))
labels = np.zeros(shape=n)

#np.random.seed(42)

km = KMeans()
centroids = km.kmeanspp_init(vectors, k)

plot2d(vectors, labels, centroids, k, f'it = -1')

for it in range(10):
    labels, centroids = km.run(vectors, centroids, labels, threshold=1e-10, iterations=10)
    plot2d(vectors, labels, centroids, k, f'it = {it}')