from matplotlib import pyplot as plt

def plot2d(vectors, labels, k):
    for label in range(k):
        x, y = vectors[labels == label].T
        plt.scatter(x, y, s=100, label=label)
    plt.legend()
    plt.show()
"""
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    ax.set_box_aspect(1)
    plt.grid()
"""

