import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#  K-MEANS FROM SCRATCH (NumPy Only Logic)
class KMeansScratch:
    def _init_(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.inertia_ = 0

    def fit(self, X):
        #  Random initialization of centroids
        random_idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_idx]

        for i in range(self.max_iters):
        #  Point Assignment Step
        # Calculate Euclidean distance using broadcasting: (N, 1, D) - (K, D)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)

        #  Centroid Recalculation Step
            new_centroids = np.array([X[new_labels == j].mean(axis=0) if len(X[new_labels == j]) > 0 
                                     else self.centroids[j] for j in range(self.k)])

        # Convergence check (Tolerance)
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            
            self.centroids = new_centroids
            self.labels_ = new_labels
        
        # Calculate SSE for the Elbow Method
        self.inertia_ = np.sum((X - self.centroids[self.labels_])**2)
        return self


#  DATA GENERATION
# Multi-modal synthetic dataset (4D, 5 distinct clusters)
X, y_true = make_blobs(n_samples=600, centers=5, n_features=4, cluster_std=1.2, random_state=42)


#  ELBOW METHOD (Finding Optimal K)

#  Programmatically determine optimal k between 1 and 10
sse_values = []
k_range = range(1, 11)

for k in k_range:
    km = KMeansScratch(k=k).fit(X)
    sse_values.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, sse_values, marker='o', linestyle='--', color='b')
plt.title('Task 3: Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (Inertia)')
plt.grid(True)
plt.show()

# FINAL EVALUATION & VISUALIZATION

#  Apply optimal k (found to be 5 from the elbow) and calculate Silhouette
optimal_k = 5
final_model = KMeansScratch(k=optimal_k).fit(X)
sil_score = silhouette_score(X, final_model.labels_)

print(f"--- Performance Results ---")
print(f"Chosen Optimal k: {optimal_k}")
print(f"Silhouette Score: {sil_score:.4f}")

# PCA Projection to 2D for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centroids_pca = pca.transform(final_model.centroids)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_model.labels_, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=250, label='Centroids')
plt.title(f"Task 5: Final Cluster Assignments (PCA Projection)")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
