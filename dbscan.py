import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def plot_k_distance_graph(data_array, k=3):
    """
    Parameters:
        data_array (np.ndarray): Data clustering dalam bentuk array.
        k (int): Jumlah tetangga terdekat yang akan dihitung.

    Returns:
        str: Base64 string dari plot.
    """
    # Hitung k-distance menggunakan NearestNeighbors
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    neighbors = nearest_neighbors.fit(data_array)
    distances, _ = neighbors.kneighbors(data_array)
    distances = np.sort(distances[:, -1])  # Ambil jarak terjauh (k-distance)

    # Buat plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'K-Distance Graph (k={k})')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k}th Nearest Neighbor Distance')
    plt.grid(True)

    # Simpan plot sebagai base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_base64

def find_optimal_dbscan_params(data, eps_range, min_pts_range):
    results = []

    for eps in eps_range:
        for min_pts in min_pts_range:
            try:
                # Inisialisasi DBSCAN dengan parameter saat ini
                dbscan = DBSCAN(eps=eps, min_pts=min_pts)
                labels = dbscan.fit(data)

                # Evaluasi hasil clustering hanya jika terdapat lebih dari 1 klaster
                if len(set(labels)) > 1:  # Jika terdapat lebih dari 1 klaster (selain outlier)
                    silhouette = silhouette_score(data, labels) if -1 in labels else 0
                    dbi = davies_bouldin_score(data, labels)
                    chi = calinski_harabasz_score(data, labels)
                    # sse = calculate_sse(data, labels)
                else:
                    silhouette = -1
                    dbi = -1
                    chi = -1
                    # sse = -1

                # Simpan hasil evaluasi
                results.append({
                    'eps': eps,
                    'min_pts': min_pts,
                    'silhouette_score': silhouette,
                    'davies_bouldin_index': dbi,
                    'calinski_harabasz_index': chi,
                    # 'sum_squared_error': sse
                })

            except Exception as e:
                # Tangani error jika ada (misalnya data tidak valid untuk parameter tertentu)
                print(f"Error with eps={eps}, min_pts={min_pts}: {e}")
                continue

    # Urutkan hasil berdasarkan kriteria evaluasi (misalnya Silhouette Score yang tertinggi)
    results = sorted(results, key=lambda x: (-x['silhouette_score'], x['davies_bouldin_index']))

    return results

def calculate_sse(data, labels):
    unique_labels = set(labels)
    sse = 0
    for label in unique_labels:
        if label == -1:  # Abaikan noise (label -1)
            continue
        cluster_points = data[labels == label]
        centroid = cluster_points.mean(axis=0)
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse

def evaluate_clustering_dbscan(data, labels):
    if len(set(labels)) > 1:  # Jika terdapat lebih dari 1 klaster (selain outlier)
        silhouette = silhouette_score(data, labels) if -1 in labels else 0
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
    else:
        silhouette = -1
        davies_bouldin = -1
        calinski_harabasz = -1

    return {
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "calinski_harabasz_index": calinski_harabasz
    }

class DBSCAN:
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts
        self.labels = []

    def fit(self, data):
        self.labels = [0] * len(data)
        cluster_id = 0

        for point_idx in range(len(data)):
            if self.labels[point_idx] != 0:  # Already processed
                continue
            neighbors = self.region_query(data, point_idx)
            if len(neighbors) < self.min_pts:
                self.labels[point_idx] = -1  # Mark as noise
            else:
                cluster_id += 1
                self.expand_cluster(data, point_idx, neighbors, cluster_id)
        return self.labels

    def expand_cluster(self, data, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if self.labels[neighbor_idx] == -1:  # Previously marked noise
                self.labels[neighbor_idx] = cluster_id
            elif self.labels[neighbor_idx] == 0:
                self.labels[neighbor_idx] = cluster_id
                new_neighbors = self.region_query(data, neighbor_idx)
                if len(new_neighbors) >= self.min_pts:
                    neighbors += new_neighbors
            i += 1

    def region_query(self, data, point_idx):
        neighbors = []
        for i in range(len(data)):
            if np.linalg.norm(data[point_idx] - data[i]) <= self.eps:
                neighbors.append(i)
        return neighbors