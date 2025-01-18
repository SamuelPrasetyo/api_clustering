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
    """
        Fungsi untuk mencari parameter optimal (eps dan min_pts) untuk algoritma DBSCAN
        berdasarkan evaluasi metrik clustering.

        Parameter:
        data : Dataset yang akan di-cluster.
        eps_range : Rentang nilai untuk parameter eps (radius lingkungan).
        min_pts_range : Rentang nilai untuk parameter min_pts (minimum titik untuk membentuk cluster).

        Mengembalikan:
        list: Daftar hasil evaluasi dengan kombinasi parameter dan metrik clustering.
    """
    results = []

    for eps in eps_range:
        for min_pts in min_pts_range:
            try:
                # Inisialisasi DBSCAN dengan parameter saat ini
                dbscan = DBSCAN(eps=eps, min_pts=min_pts)
                labels = dbscan.fit(data)

                # Hitung jumlah cluster (label unik, kecuali label -1 untuk outlier)
                unique_labels = set(labels)
                num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

                # Evaluasi hasil clustering
                if num_clusters > 1:  # Hanya hitung metrik jika ada lebih dari 1 cluster
                    silhouette = silhouette_score(data, labels) if num_clusters > 1 else -1
                    dbi = davies_bouldin_score(data, labels)
                    chi = calinski_harabasz_score(data, labels)
                else:
                    silhouette = -1
                    dbi = -1
                    chi = -1

                # Simpan hasil evaluasi
                results.append({
                    'eps': eps,
                    'min_pts': min_pts,
                    'num_clusters': num_clusters,
                    'silhouette_score': silhouette,
                    'davies_bouldin_index': dbi,
                    'calinski_harabasz_index': chi
                })

            except Exception as e:
                # Tangani error jika ada (data tidak valid untuk parameter tertentu)
                print(f"Error with eps={eps}, min_pts={min_pts}: {e}")
                continue

    # Urutkan hasil berdasarkan kriteria evaluasi (Silhouette Score yang tertinggi)
    results = sorted(results, key=lambda x: (-x['silhouette_score'], x['davies_bouldin_index']))

    return results

# def find_optimal_dbscan_params(data, eps_range, min_pts_range):
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
                else:
                    silhouette = -1
                    dbi = -1
                    chi = -1

                # Simpan hasil evaluasi
                results.append({
                    'eps': eps,
                    'min_pts': min_pts,
                    'silhouette_score': silhouette,
                    'davies_bouldin_index': dbi,
                    'calinski_harabasz_index': chi
                })

            except Exception as e:
                # Tangani error jika ada (misalnya data tidak valid untuk parameter tertentu)
                print(f"Error with eps={eps}, min_pts={min_pts}: {e}")
                continue

    # Urutkan hasil berdasarkan kriteria evaluasi (Silhouette Score yang tertinggi)
    results = sorted(results, key=lambda x: (-x['silhouette_score'], x['davies_bouldin_index']))

    return results

def calculate_centroids(data, labels):
    """
        Menghitung centroid dari setiap kluster yang dihasilkan oleh DBSCAN.

        Parameters:
            data (np.ndarray): Data clustering dalam bentuk array (dimensi 2).
            labels (list or np.ndarray): Label hasil clustering DBSCAN untuk setiap titik data.

        Returns:
            dict: Dictionary yang berisi centroid untuk setiap kluster dalam format {cluster_id: centroid_array}.
    """
    unique_labels = set(labels)
    centroids = {}

    for label in unique_labels:
        if label == -1:  # Abaikan noise (label -1)
            continue

        # Ambil data untuk kluster tertentu
        cluster_points = data[np.array(labels) == label]

        # Hitung centroid sebagai rata-rata dari semua titik dalam kluster
        centroid = np.mean(cluster_points, axis=0)

        # Simpan centroid ke dalam dictionary
        centroids[label] = centroid

    return centroids

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
        """
            Inisialisasi algoritma DBSCAN.
            
            Parameter:
            eps (float): Jarak maksimum antara dua titik agar dianggap sebagai tetangga.
            min_pts (int): Jumlah minimum titik yang diperlukan untuk membentuk daerah padat (core point).
        """
        self.eps = eps
        self.min_pts = min_pts
        self.labels = []

    def fit(self, data):
        """
            Terapkan clustering DBSCAN pada data yang diberikan.

            Parameter:
            data (list atau ndarray): Dataset yang akan dikelompokkan.

            Mengembalikan:
            list: Label cluster untuk setiap titik dalam data.
        """
        self.labels = [0] * len(data)
        cluster_id = 0

        for point_idx in range(len(data)):
            if self.labels[point_idx] != 0:
                continue
            neighbors = self.region_query(data, point_idx)
            if len(neighbors) < self.min_pts:
                self.labels[point_idx] = -1  # Tandai titik sebagai noise (label = -1)
            else:
                cluster_id += 1 # Tambahkan ID cluster untuk cluster baru
                # Perluas cluster dari titik saat ini
                self.expand_cluster(data, point_idx, neighbors, cluster_id)
        return self.labels

    def expand_cluster(self, data, point_idx, neighbors, cluster_id):
        """
            Perluas cluster secara rekursif dengan menambahkan semua titik yang dapat dijangkau.

            Parameter:
            data (list atau ndarray): Dataset yang akan dikelompokkan.
            point_idx (int): Indeks titik awal untuk cluster.
            neighbors (list): Daftar tetangga dari titik awal.
            cluster_id (int): ID cluster saat ini.
        """
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if self.labels[neighbor_idx] == -1: # Jika sebelumnya ditandai sebagai noise
                self.labels[neighbor_idx] = cluster_id
            elif self.labels[neighbor_idx] == 0: # Jika titik belum diproses
                self.labels[neighbor_idx] = cluster_id
                # Cari tetangga dari titik tetangga
                new_neighbors = self.region_query(data, neighbor_idx)
                if len(new_neighbors) >= self.min_pts:
                    neighbors += new_neighbors # Tambahkan tetangga baru ke daftar
            i += 1

    def region_query(self, data, point_idx):
        """
            Cari semua titik di lingkungan sebuah titik berdasarkan jarak eps.

            Parameter:
            data (list atau ndarray): Dataset yang akan dikelompokkan.
            point_idx (int): Indeks titik yang sedang diperiksa.

            Mengembalikan:
            list: Indeks titik-titik dalam radius lingkungan.
        """
        neighbors = []
        for i in range(len(data)):
            # Hitung jarak Euclidean antara titik saat ini dan titik lainnya
            if np.linalg.norm(data[point_idx] - data[i]) <= self.eps:
                neighbors.append(i)
        return neighbors