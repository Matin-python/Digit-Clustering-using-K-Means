import cv2 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans 
from sklearn import datasets
from sklearn.metrics import accuracy_score

from scipy.stats import mode


digits = datasets.load_digits()

print("=" * 50)
print("digits.data.shape = ", digits.data.shape)
print("digits.images.shape = ", digits.images.shape)
print("digits.target.shape = ", digits.target.shape)
print("=" * 50)

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
plt.suptitle("K-Means Cluster Centers")
plt.tight_layout()
plt.show()

print(labels)

print(accuracy_score(digits.target, labels))

cluster_labels = np.zeros_like(clusters)
for cluster_number in range (10):
    mask = (clusters == cluster_number)
    cluster_labels[mask] = mode(clusters[mask], keepdims=True).mode[0]

image = []
for i in range (10):
    image.append(cv2.imread(f'test dataset\{i}.jpg'))
    if image is None:
        raise FileNotFoundError(f"Could not load test dataset\\{i}.jpg")
    
    image[i] = cv2.cvtColor(image[i], cv2.COLOR_BGR2RGB)  # turn BGR to RGB
    image[i] = image[i][:, :, 0]
    image[i] = image[i].flatten()

image = np.array(image)

predicted_clusters = kmeans.predict(image)
print()
print("Predicted Clusters = ")
print(predicted_clusters)

real_labels = np.arange(10)
predicted_labels = np.zeros_like(predicted_clusters)
for cluster_number in range(10):
    mask = predicted_clusters == cluster_number
    if np.any(mask):
        predicted_labels[mask] = mode(real_labels[mask], keepdims=True).mode[0]

custom_accuracy = accuracy_score(real_labels, predicted_labels)

print()
print("Real Labels = ")
print(real_labels)

print("Predicted Labels = ")
print(predicted_labels)

print()
print("Custom Image Accuracy = ", custom_accuracy * 100, "%")

plt.show()
