from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# # Fungsi memproses file excel
# def process_file(file_path):
#     # Membaca file Excel
#     df = pd.read_excel(file_path)
#
#     # Konversi kolom nilai menjadi numerik
#     for column in ['Nilai1', 'Nilai2', 'Nilai3', 'Nilai4']:
#         df[column] = pd.to_numeric(df[column], errors='coerce')  # Mengubah yang gagal jadi NaN
#
#     # Cek apakah ada nilai yang tidak valid setelah konversi
#     if df.isnull().any().any():
#         return "Error: Beberapa nilai tidak valid (mungkin ada teks yang tidak bisa diubah menjadi angka)."
#
#     return df
# Fungsi memproses file excel
def process_file(file_path):
    # Membaca file Excel
    df = pd.read_excel(file_path)

    # Ambil kolom dari kolom D ke depan
    selected_columns = df.columns[3:]  # Kolom D adalah indeks ke-3 (dimulai dari 0)

    # Konversi kolom nilai menjadi numerik
    for column in selected_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Mengubah yang gagal jadi NaN

    # Cek apakah ada nilai yang tidak valid setelah konversi
    if df[selected_columns].isnull().any().any():
        return "Error: Beberapa nilai tidak valid (mungkin ada teks yang tidak bisa diubah menjadi angka)."

    # Kembalikan DataFrame dengan hanya kolom yang dipilih
    return df[selected_columns]

# Fungsi Euclidean Distance
def euclidean_distance(point, centroids):
    return np.sqrt(np.sum((point - centroids) ** 2, axis=1))

# Fungsi K-Means
def k_means_clustering(data, k, max_iters=100):
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
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

# Fungsi Elbow Method
def find_optimal_k(data, max_k=10):
    distortions = []
    for k in range(1, max_k + 1):
        centroids, clusters = k_means_clustering(data, k)
        intra_cluster_distances = np.min(cdist(data, centroids, 'euclidean'), axis=1)
        distortions.append(np.sum(intra_cluster_distances ** 2))
    return distortions

# Fungsi Davies-Bouldin Index (DBI)
def davies_bouldin_index(data, clusters, centroids):
    n_clusters = len(np.unique(clusters))
    db_index = 0
    for i in range(n_clusters):
        cluster_points = data[clusters == i]
        if len(cluster_points) == 0:
            continue
        s_i = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))
        max_ratio = 0
        for j in range(n_clusters):
            if i == j:
                continue
            cluster_points_j = data[clusters == j]
            if len(cluster_points_j) == 0:
                continue
            s_j = np.mean(np.linalg.norm(cluster_points_j - centroids[j], axis=1))
            m_ij = np.linalg.norm(centroids[i] - centroids[j])
            max_ratio = max(max_ratio, (s_i + s_j) / m_ij)
        db_index += max_ratio
    db_index /= n_clusters
    return db_index

# API Endpoint
@app.route('/kmeans', methods=['POST'])
def kmeans():
    try:
        # Ambil file yang dikirim dari request
        file = request.files['file']

        # Simpan file sementara di server
        file_path = "temp_file.xlsx"
        file.save(file_path)

        # Proses file
        df = process_file(file_path)

        # Cek jika ada error dalam data
        if isinstance(df, str):  # Jika hasilnya adalah pesan error
            return jsonify({'error': df}), 400

        # Ambil data nilai (kolom Nilai1, Nilai2, Nilai3, Nilai4)
        data = df.values

        # Tentukan nilai k terbaik menggunakan Elbow Method
        distortions = find_optimal_k(data)
        # optimal_k = distortions.index(min(distortions)) + 1
        optimal_k = 4

        # Jalankan K-Means dengan nilai k terbaik
        centroids, clusters = k_means_clustering(data, optimal_k)

        # Tambahkan hasil cluster ke DataFrame
        df['Cluster'] = clusters

        # Evaluasi hasil clustering (misalnya menggunakan Davies-Bouldin Index)
        db_index = davies_bouldin_index(data, clusters, centroids)

        # Kembalikan hasil
        return jsonify({
            "optimal_k": optimal_k,
            "davies_bouldin_index": db_index,
            "final_centroids": centroids.tolist(),
            "data": df.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400



# def kmeans_api():
#     # Terima file Excel
#     file = request.files.get('file')
#     if not file:
#         return jsonify({"error": "No file uploaded"}), 400
#
#     # Baca file Excel
#     df = pd.read_excel(file)
#
#     # Ambil kolom nilai
#     data = df.iloc[:, 2:].values  # Asumsi kolom pertama adalah ID, kedua adalah Nama
#
#     # Tentukan nilai k terbaik menggunakan Elbow
#     distortions = find_optimal_k(data)
#     optimal_k = distortions.index(min(distortions)) + 1
#
#     # Jalankan K-Means dengan nilai k terbaik
#     centroids, clusters = k_means_clustering(data, optimal_k)
#
#     # Evaluasi hasil clustering
#     db_index = davies_bouldin_index(data, clusters, centroids)
#
#     # Tambahkan hasil cluster ke DataFrame
#     df['Cluster'] = clusters
#
#     # Simpan hasil clustering ke file baru
#     result_file = "hasil_clustering.xlsx"
#     df.to_excel(result_file, index=False)
#
#     # Kembalikan hasil
#     return jsonify({
#         "optimal_k": optimal_k,
#         "davies_bouldin_index": db_index,
#         "result_file": result_file
#     })

if __name__ == '__main__':
    app.run(debug=True)
