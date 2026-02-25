# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset and print head,info of the dataset

2.Check for null values

3.Import kmeans and fit it to the dataset

4.Plot the graph using elbow method

5.Print the predicted array

6.Plot the customer segments 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Sahana S
RegisterNumber: 25013621
*/
```
~~~
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 2: Load Dataset
dataset = pd.read_csv('/mnt/data/Mall_Customers.csv')

# Display first 5 rows
print(dataset.head())

# Step 3: Select Features (Annual Income & Spending Score)
X = dataset.iloc[:, [3, 4]].values

# Step 4: Elbow Method to find optimal K
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

print("WCSS Values:")
print(wcss)

# Plot Elbow Graph
plt.figure()
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Step 5: Apply KMeans with optimal K (usually 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

print("Cluster Labels:")
print(y_kmeans)

# Step 6: Visualize Clusters
plt.figure()

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50)
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50)
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50)
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50)
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50)

# Plot Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=200, marker='X')

~~~

## Output:
<img width="722" height="149" alt="image" src="https://github.com/user-attachments/assets/64842b96-d5ee-47f7-8eb6-2eb5bf8d92ce" />
<img width="865" height="573" alt="image" src="https://github.com/user-attachments/assets/d140ce73-a9cb-4174-b264-b13ebe5a9a11" />
<img width="747" height="158" alt="image" src="https://github.com/user-attachments/assets/180dd363-510a-4f6d-b490-ef6dbb19e33b" />
<img width="739" height="542" alt="image" src="https://github.com/user-attachments/assets/457722ca-5e3f-44fb-bf42-60afe44b6b4f" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
