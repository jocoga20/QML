import numpy as np
from random import choice
from kmeans import KMeans

class DeltaKMeans(KMeans):
    def __init__(self, delta, seed=42):
        super().__init__()
        self.delta = delta
        self.seed = seed
    
    def assign_label(self, vector, centroids):
        k = centroids.shape[0]
        d_min = float('inf')
        filtered_centroids = []
        distances = []

        for i in range(k):
            d = np.linalg.norm(vector - centroids[i])
            distances.append(d)
            print(d)

            if d > d_min + self.delta:
                continue

            if d < d_min:
                d_min = d

            filtered_centroids.append(i)
        
        filtered_centroids = [i for i in filtered_centroids if distances[i] <= d_min + self.delta]
        return choice(filtered_centroids)

    def assign_labels(self, vectors, centroids):
        return np.array([self.assign_label(v, centroids, self.delta) for v in vectors])
    
    def update_centroids(self, vectors, labels, k):
        d = vectors.shape[1]
        return super().update_centroids(vectors, labels, k) + self.delta / (2 * d)
    