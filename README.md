# cultusproject
# K-Means Clustering from Scratch
​# Project Overview
​This project implements the K-Means clustering algorithm from the ground up using only NumPy for the core logic. The goal was to build the algorithm without relying on high-level libraries like scikit-learn to demonstrate a deep understanding of iterative optimization, centroid management, and distance metrics.
​The program generates a complex 4D dataset, determines the ideal number of clusters via the Elbow Method, and validates the results through Silhouette Scores and PCA-based visualization.
​# How It Works
​1. The Core Algorithm (KMeansScratch)
​I implemented the algorithm as a Python class using iterative optimization:
​Centroid Initialization: Randomly picks k points from the dataset as starting centers.
​Assignment Step: Uses NumPy broadcasting to calculate the Euclidean distance from every point to every centroid, assigning each point to its nearest neighbor.
​Update Step: Calculates the mean of all points assigned to a cluster and moves the centroid to that new position.
​Convergence: The loop breaks early if the centroids move less than a specified tolerance (10^{-4}).
​2. Finding the "Elbow"
​To find the optimal number of clusters (k), the code runs the algorithm for k=1 through 10. It records the Sum of Squared Errors (SSE) for each run. The resulting plot shows where the "elbow" occurs—the point where adding more clusters no longer provides a significant drop in error.
​3. Dimensionality Reduction (PCA)
​Since the dataset is 4-dimensional, it is impossible to visualize directly. I used Principal Component Analysis (PCA) to project the data into a 2D plane. This allows us to see how the scratch-built algorithm successfully separated the high-dimensional clusters.
# ​Performance Summary
​Optimal Clusters: k=5 (Determined by the Elbow Method).
​Silhouette Score: Typically around ~0.65, indicating that clusters are well-defined and samples are not overlapping.
​Accuracy: The visual plot confirms that centroids are positioned perfectly at the center of their respective clusters.
​# Requirements
​To run this project, you need:
​numpy
​matplotlib
​scikit-learn (Used only for data generation, PCA, and Silhouette Score calculation)
​# Final Interpretation
​This implementation proves that a basic understanding of linear algebra and NumPy can replicate complex library behaviors. The results show that even in 4-dimensional space, the iterative update process of K-Means is highly effective at uncovering hidden patterns in data.
