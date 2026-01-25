import numpy as np
from sklearn.decomposition import PCA
from torchvision import transforms, datasets
from deltakmeans import DeltaKMeans
from kmeans import KMeans
from utils import accuracy, compute_centroids_classes, vectors_to_class

def get_mnist(train):
    m = datasets.MNIST('./data', train=train, transform=transforms.ToTensor(), download=True)
    return m.data.numpy().reshape(-1, 784), m.targets.numpy()

def prepare_dataset(kmeans, k, d):
    X_tr, y_tr = get_mnist(True)
    X_ts, y_ts = get_mnist(False)

    print('PCA')
    pca = PCA(d)
    X_tr = pca.fit_transform(X_tr)
    X_ts = pca.transform(X_ts)

    print('Scaling')
    min_norm = np.linalg.norm(X_tr, axis=1).min()
    X_tr = X_tr / min_norm
    X_ts = X_ts / min_norm

    labels = np.zeros(shape=X_tr.shape)

    print('kmeans++ init')
    centroids = kmeans.kmeanspp_init(X_tr, k)
    return X_tr, y_tr, X_ts, y_ts, labels, centroids

def run(kmeans, k, d, full_ds):
    X_tr, y_tr, X_ts, y_ts, labels, centroids = full_ds
    print(kmeans.__class__)
    labels, centroids = kmeans.run(vectors=X_tr, centroids=centroids, labels=labels, max_it=100, threshold=1e-4)
    centroid_to_class = compute_centroids_classes(k, labels, y_tr)
#    print('Mapping', centroid_to_class)
    y_pred_tr = np.array([centroid_to_class[l] for l in labels])
    y_pred_ts = vectors_to_class(X_ts, centroids, centroid_to_class)

    return accuracy(y_pred_tr, y_tr).item(), accuracy(y_pred_ts, y_ts).item()

full_ds = prepare_dataset(KMeans(), k=20, d=40)
print('KMEANS')
for k1 in [20, 40, 60]:
    print(k1)
    print(run(KMeans(), k=k1, d=40, full_ds=full_ds))

print('DELTA KMEANS')
for k1 in [20, 40, 60]:
    for delta in [0.1, 0.25, 0.5]:
        print(delta, run(DeltaKMeans(delta), k=20, d=40, full_ds=full_ds))
