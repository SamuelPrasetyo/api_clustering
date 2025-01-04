import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Import Plot
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64



""" Fungsi perhitungan Euclidean Distance """
def euclidean_distance(point, centroids):
    return np.sqrt(np.sum((point - centroids) ** 2, axis=1))



""" Inisialisasi centroid dengan K-Means++ """
def initialize_centroids(data, k):
    np.random.seed(42)
    n_samples = data.shape[0]
    centroids = []

    # Pilih pusat pertama secara acak
    centroids.append(data[np.random.randint(0, n_samples)])

    # Pilih sisa pusat dengan probabilitas proporsional terhadap jarak
    for _ in range(1, k):
        distances = np.min(cdist(data, np.array(centroids), metric="euclidean"), axis=1)
        probabilities = distances ** 2 / np.sum(distances ** 2)
        new_centroid_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(data[new_centroid_idx])

    return np.array(centroids)



""" Fungsi K-Means """
def k_means_clustering(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = []
        for point in data:
            distances = euclidean_distance(point, centroids)
            cluster = np.argmin(distances)
            clusters.append(cluster)
        clusters = np.array(clusters)

        new_centroids = []
        for i in range(k):
            cluster_points = data[clusters == i]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[i])
        new_centroids = np.array(new_centroids)

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, clusters



""" Fungsi Evaluasi Clustering """
def evaluate_clustering_kmeans(data, clusters, centroids):
    silhouette_avg = silhouette_score(data, clusters)
    davies_bouldin = davies_bouldin_score(data, clusters)
    calinski_harabasz = calinski_harabasz_score(data, clusters)

    # Hitung Sum of Squared Error (SSE)
    sse = np.sum(np.min(cdist(data, centroids, 'euclidean') ** 2, axis=1))

    return {
        "silhouette_score": silhouette_avg,
        "davies_bouldin_index": davies_bouldin,
        "calinski_harabasz_index": calinski_harabasz,
        "sum_squared_error": sse
    }



""" Fungsi Elbow Method Sklearn """
def find_optimal_k_sklearn(data, max_k=10):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)  # inertia_ adalah sum of squared distances (distorsi)
    
    # Visualisasi elbow curve dan simpan sebagai gambar
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url, distortions

# Fungsi Elbow Method Scratch
# def find_optimal_k(data, max_k=10):
#     distortions = []
#     for k in range(1, max_k + 1):
#         centroids, clusters = k_means_clustering(data, k)
#         intra_cluster_distances = np.min(cdist(data, centroids, 'euclidean'), axis=1)
#         distortions.append(np.sum(intra_cluster_distances ** 2))
#     return distortions