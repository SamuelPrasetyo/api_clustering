import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def euclidean_distance(point1, point2):
    """Menghitung jarak Euclidean antara dua titik."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_cluster_distance(cluster1, cluster2):
    """Menghitung jarak rata-rata antara dua cluster."""
    distances = [
        euclidean_distance(point1, point2)
        for point1 in cluster1
        for point2 in cluster2
    ]
    return np.mean(distances)

def agglomerative_clustering(data, n_clusters):
    """Implementasi Agglomerative Clustering."""
    # Awalnya, setiap data adalah cluster
    clusters = [[point] for point in data]

    while len(clusters) > n_clusters:
        min_distance = float('inf')
        merge_pair = (None, None)

        # Cari dua cluster dengan jarak terdekat
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = calculate_cluster_distance(clusters[i], clusters[j])
                if dist < min_distance:
                    min_distance = dist
                    merge_pair = (i, j)

        # Gabungkan dua cluster terdekat
        cluster1, cluster2 = merge_pair
        clusters[cluster1].extend(clusters[cluster2])
        clusters.pop(cluster2)

    # Labelkan data berdasarkan cluster
    labels = np.zeros(len(data), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for point in cluster:
            index = np.where((data == point).all(axis=1))[0][0]
            labels[index] = cluster_id

    return labels

def evaluate_clustering_agglomerative(clustering_data, labels):
    """Evaluasi hasil clustering dengan berbagai metrik."""
    db_index = davies_bouldin_score(clustering_data, labels)
    silhouette = silhouette_score(clustering_data, labels)
    calinski_harabasz = calinski_harabasz_score(clustering_data, labels)

    return {
        'davies_bouldin_index': db_index,
        'silhouette_score': silhouette,
        'calinski_harabasz_index': calinski_harabasz
    }
    
def calculate_centroids(clustering_data, labels):
    """Menghitung centroid setiap cluster."""
    centroids = {}
    unique_labels = np.unique(labels)
    for cluster_id in unique_labels:
        cluster_points = clustering_data[labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids[cluster_id] = centroid

    return {int(k): list(v) for k, v in centroids.items()}

def calculate_sse(clustering_data, labels, n_clusters):
    """Menghitung Sum of Squared Error (SSE)."""
    centers = [clustering_data.values[labels == i].mean(axis=0) for i in range(n_clusters)]
    sse = sum(np.linalg.norm(clustering_data.values - centers[label]) ** 2 for label, centers in zip(labels, centers))
    return sse