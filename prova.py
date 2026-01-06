from sklearn.decomposition import PCA
from torchvision import transforms, datasets

from kmeans import KMeans

X = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True).data.numpy()
X = X.reshape(-1, 784)

km = KMeans()

km.run(X, )