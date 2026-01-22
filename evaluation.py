import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import transforms, datasets
from deltakmeans import DeltaKMeans
from kmeans import KMeans
from utils import accuracy, compute_centroids_classes, vectors_to_class

def get_mnist(train):
    m = datasets.MNIST('./data', train=train, transform=transforms.ToTensor(), download=True)
    return m.data.numpy().reshape(-1, 784), m.targets.numpy()

def run(kmeans, k, d):
    X_tr, y_tr = get_mnist(True)
    X_ts, y_ts = get_mnist(False)

    print('Scaling')
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ts = scaler.transform(X_ts)

    print('PCA')
    pca = PCA(d)
    X_tr = pca.fit_transform(X_tr)
    X_ts = pca.transform(X_ts)

    labels = np.zeros(shape=X_tr.shape)

    print('kmeans++ init')
    init_centroids_path = f'inits/mnist.k{k}.d{d}.init.npy'

    try:
        print('Loading kmeans++ init centroids')
        centroids = np.load(init_centroids_path, allow_pickle=False)
    except:
        centroids = kmeans.kmeanspp_init(vectors=X_tr, k=k)
        print('Saving kmeans++ init centroids')
        np.save(init_centroids_path, centroids, allow_pickle=False)

    print(kmeans.__class__)
    labels, centroids = kmeans.run(vectors=X_tr, centroids=centroids, labels=labels, max_it=100, threshold=1e-4)
    centroid_to_class = compute_centroids_classes(k, labels, y_tr)
#    print('Mapping', centroid_to_class)
    y_pred_tr = np.array([centroid_to_class[l] for l in labels])
    y_pred_ts = vectors_to_class(X_ts, centroids, centroid_to_class)

    return accuracy(y_pred_tr, y_tr), accuracy(y_pred_ts, y_ts)

with open('results.txt', 'w') as f:
    for delta in [0.1, 0.5]:
        for k in [20, 40, 80]:
            if delta == 0.1 and k == 20:
                continue
            tr_acc, ts_acc = run(DeltaKMeans(delta=delta), k=k, d=40)
            f.write(f'- & {k} & {tr_acc} & {ts_acc}\\\n')