# 🔢 Handwritten Digit Clustering using K-Means

An **unsupervised machine learning** project that uses the **K-Means clustering algorithm** to group handwritten digit images from the **scikit-learn Digits dataset**.

The project demonstrates how K-Means can discover groups of handwritten digits **without using their labels during training**, and then maps the resulting clusters to digit classes for evaluation.

It also tests the trained clustering model on custom handwritten digit images.


## Overview

This project uses **K-Means clustering** to group the handwritten digits from 0 to 9 into **10 clusters**.

Unlike supervised learning algorithms such as Logistic Regression, K-Means does not use the target labels when creating the clusters.

After clustering, the project uses the most common actual digit within each cluster to assign a digit label to that cluster. The assigned labels are then compared with the real labels to measure the clustering performance.

The project also uses **10 custom handwritten digit images** stored in the `test dataset` folder and predicts their corresponding clusters.


## Features

* 🤖 K-Means clustering
* 🔢 Clustering handwritten digits from 0–9
* 📊 Uses the scikit-learn Digits dataset
* 🧠 Unsupervised learning
* 🔟 Creates 10 clusters
* 🖼️ Visualizes the cluster centers
* 🎯 Maps clusters to digit labels using the most common label
* 📈 Calculates clustering accuracy
* 🖼️ Predicts custom handwritten digit images
* 📊 Compares predicted and real outputs


## Technologies Used

* Python 3
* NumPy
* OpenCV
* Scikit-learn
* Matplotlib
* SciPy


## Dataset

The project uses the **Digits Dataset** provided by scikit-learn.

The dataset contains:

* **1,797 handwritten digit samples**
* **10 classes:** 0–9
* **64 features per sample**
* Each image has a resolution of **8 × 8 pixels**
* Grayscale pixel values

Each 8×8 image is represented as a 64-dimensional feature vector when training the K-Means model.


## Machine Learning Workflow

The project follows these main steps:

1. Load the Digits dataset from scikit-learn.
2. Create a K-Means model with **10 clusters**.
3. Train the model using the image data.
4. Obtain the cluster assigned to each digit.
5. Find the most common actual digit in each cluster.
6. Use these values to assign digit labels to the clusters.
7. Calculate the clustering accuracy.
8. Visualize the 10 cluster centers.
9. Load 10 custom handwritten digit images.
10. Convert the images into the required format.
11. Predict their clusters using the trained K-Means model.
12. Map the predicted clusters to digit labels.
13. Calculate the accuracy on the custom images.


## K-Means Clustering

The model is created with:

```python
kmeans = KMeans(n_clusters=10, random_state=0)
```

The number of clusters is set to **10** because the dataset contains ten different digits, from 0 to 9.

The model learns the clusters using:

```python
clusters = kmeans.fit_predict(digits.data)
```

The actual digit labels are **not used during this training process**.


## Cluster Visualization

After training, the cluster centers are reshaped from 64-dimensional vectors back into 8×8 images:

```python
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
```

The project then displays the ten cluster centers as images.

This provides a visual representation of what each cluster has learned.


## Cluster Label Mapping

Because K-Means does not know that a particular cluster represents a specific digit, the clusters need to be mapped to digit labels.

The project finds the most common actual digit inside each cluster:

```python
labels[mask] = mode(digits.target[mask])[0]
```

For example, if most of the samples assigned to cluster `3` are actually digit `7`, that cluster is considered to represent digit `7`.

This allows the clustering results to be compared with the actual digit labels.


## Custom Handwritten Digits

The project also supports custom handwritten digit images.

The images are stored in:

```text
test dataset/
```

The expected files are:

```text
test dataset/
├── 0.jpg
├── 1.jpg
├── 2.jpg
├── 3.jpg
├── 4.jpg
├── 5.jpg
├── 6.jpg
├── 7.jpg
├── 8.jpg
└── 9.jpg
```

Each image is:

1. Loaded using OpenCV.
2. Converted from BGR to RGB.
3. Converted to a single channel.
4. Flattened into a 64-element feature vector.
5. Passed to the trained K-Means model.

The predicted clusters are then mapped to digit labels.


## Project Structure

```text
Digit-Clustering-using-K-Means/
│
├── digit_clustering.py
├── test dataset/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── 7.jpg
│   ├── 8.jpg
│   └── 9.jpg
├── requirements.txt
├── LICENSE
└── README.md
```


## Installation

Clone the repository:

```bash
git clone https://github.com/Matin-python/Handwritten-Digit-Clustering.git
```

Move into the project directory:

```bash
cd Handwritten-Digit-Clustering
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install numpy opencv-python scikit-learn matplotlib scipy
```


## How to Run

Run the Python script:

```bash
python digit_clustering.py
```

The program will:

* Load the Digits dataset.
* Train the K-Means model.
* Display the cluster centers.
* Map clusters to digit labels.
* Calculate clustering accuracy.
* Load the custom handwritten digit images.
* Predict their clusters.
* Calculate the prediction accuracy.


## Evaluation

The project calculates accuracy using:

```python
accuracy_score(digits.target, labels)
```

For the custom images, the predicted labels are compared with:

```python
real_out = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### Important Note

Although **accuracy** is reported, this is fundamentally an **unsupervised learning project**.

The target labels are only used **after clustering** to determine which digit each cluster most likely represents. They are not used to train the K-Means model.


## Example Output

The program displays the shape of the cluster centers:

```text
(10, 64)
```

It then prints the mapped labels and clustering accuracy.

Example:

```text
(10, 64)

[...cluster labels...]

0.XX
```

It also predicts the custom handwritten digits:

```text
[0 1 2 3 4 5 6 7 8 9]
```

The exact results may vary depending on the version of scikit-learn and the clustering result.


## Visualization

The project generates a visualization containing the **10 K-Means cluster centers**.

Each center represents the average pattern learned for one cluster of handwritten digits.

This makes it possible to visually inspect what the K-Means algorithm has learned from the dataset.


## Related Project

This project is related to my **Handwritten Digit Recognition using Logistic Regression** project.

### 🔢 Handwritten Digit Recognition using Logistic Regression

Unlike K-Means, Logistic Regression is a **supervised learning algorithm** that uses the actual digit labels during training.

The Logistic Regression project includes:

* Multiclass classification of digits 0–9
* Binary classification of digits 0 and 1
* Prediction of custom handwritten digit images
* Classification accuracy evaluation
* Mean Squared Error calculation

➡️ **Repository:**
[https://github.com/Matin-python/Digits-Classification-Logistic-Regression.git](https://github.com/Matin-python/Digits-Classification-Logistic-Regression.git)

### Main Difference

| Logistic Regression                | K-Means                                    |
| ---------------------------------- | ------------------------------------------ |
| Supervised learning                | Unsupervised learning                      |
| Uses target labels during training | Does not use target labels during training |
| Classification                     | Clustering                                 |
| Predicts known classes             | Discovers groups                           |
| Labels are known beforehand        | Cluster labels are assigned afterward      |


## Future Improvements

* 📊 Confusion matrix visualization
* 📈 Detailed clustering evaluation
* 🎨 Improve custom image preprocessing
* 🖼️ Add interactive digit drawing
* 📷 Real-time webcam digit recognition
* 🔄 Experiment with different numbers of clusters
* 🧠 Compare K-Means with other clustering algorithms
* 📊 Visualize cluster assignments
* 💾 Save and load the trained model
* ⚙️ Improve K-Means parameter tuning


## Contributing

Contributions, suggestions, and bug reports are welcome.

Feel free to fork the repository and submit a pull request.


## License

This project is licensed under the MIT License.


## Author

**Mohammad Reza Bakhshandeh**

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Machine Learning, Deep Learning, Computer Vision, Artificial Intelligence, and Game Development.
