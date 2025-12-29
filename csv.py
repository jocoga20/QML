import numpy as np
from plot import plot2d
from kmeans import kmeans, compute_centroids

matrix = np.genfromtxt('datasets\\city_lifestyle_dataset.csv', delimiter=',')
head = matrix[0]
print(head)
matrix = matrix[1:, (3,-2)]
k = 2
n, d = matrix.shape
labels = [i % k for i in range(n)]
centroids = compute_centroids(matrix, labels, k)

for it in range(2):
    labels, centroids = kmeans(matrix, centroids)
    plot2d(matrix, labels, k)