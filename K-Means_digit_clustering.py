import cv2 
import numpy as np 

from sklearn.cluster import KMeans 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import mode


digits = datasets.load_digits()

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape
print(kmeans.cluster_centers_.shape)

fig, ax = plt.subplots(2, 5, figsize= (8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set (xticks=[], yticks=[])
    axi.imshow(center, interpolation= 'nearest', cmap=plt.cm.binary)

labels = np.zeros_like(clusters)
for i in range (10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

print(labels)
print(accuracy_score(digits.target, labels))

plt.show()

image = []
for i in range (10):
    image.append(cv2.imread(f'test dataset\{i}.jpg'))
    image[i] = cv2.cvtColor(image[i], cv2.COLOR_BGR2RGB)  # turn BGR to RGB
    image[i] = image[i][:, :, 0]
    image[i] = image[i].flatten()

image = np.array(image)

out_predict = kmeans.predict(image)
print(out_predict)
real_out = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

labels = np.zeros_like(out_predict)
for i in range (10):
    mask = (out_predict == i)
    labels[mask] = mode(real_out[mask])[0]
print(accuracy_score(real_out, labels))
