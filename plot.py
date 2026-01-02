from matplotlib import pyplot as plt
plt.ion()
fig, ax = plt.subplots()

def plot2d(vectors, labels, centroids, k, title):
    ax.clear()
    for label in range(k):
        x, y = vectors[labels == label].T
        plt.scatter(x, y, s=100, label=label)
        x, y = centroids[label]
        plt.scatter(x, y, s=100, label=f'c({label})')
    plt.title(title)
    plt.legend()
    plt.pause(1)
"""
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    ax.set_box_aspect(1)
    plt.grid()
"""

