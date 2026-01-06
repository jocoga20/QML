import numpy as np

np.random.seed(42)

class KMeans:
    def get_probs(vectors, centroids):
        

    def kmeanspp_init(self, vectors, k):
        n = vectors.shape[0]
        i = np.random.choice(n)
        centroids = [vectors[i]]
        
        for i in range(k-1):
            self.get_probs(vectors, centroids)
    
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

km = KMeans()
x = km.kmeanspp_init(np.array([[0,0], [1,1], [2,2]]))
print(x)