import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import transforms, datasets
from deltakmeans import DeltaKMeans

def get_mnist(train):
    m = datasets.MNIST('./data', train=train, transform=transforms.ToTensor(), download=True)
    return m.data.numpy().reshape(-1, 784), m.targets.numpy()

X_tr, y_tr = get_mnist(True)
X_ts, y_ts = get_mnist(False)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_ts = scaler.transform(X_ts)

d = 40
pca = PCA(d)
X_tr = pca.fit_transform(X_tr)
X_ts = pca.transform(X_ts)

k = 10

km = DeltaKMeans(delta=.5)
c = km.kmeanspp_init(vectors=X_tr, k=k)

labels, centroids = km.run(vectors=X_tr, centroids=c, labels=np.zeros(shape=X_tr.shape[0]), max_it=100, threshold=1e-4)

def get_centroids_majority_class(labels, y, k):
    return [np.bincount(y[labels == i]).argmax() for i in range(k)]

def nearest_centroid_index(vector, centroids):
    return np.linalg.norm(centroids - vector, axis=1).argmin()

majority_class_by_centroid = get_centroids_majority_class(labels, y_tr, k)

accuracy = 0.
preds = []
for x in X_ts:
    i = nearest_centroid_index(x, centroids)
    pred = majority_class_by_centroid[i]
    preds.append(pred)

preds = np.array(preds)
accuracy = (preds == y_ts).sum() / X_ts.shape[0]

print(accuracy)