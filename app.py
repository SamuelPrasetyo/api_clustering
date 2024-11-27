from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from flask_cors import CORS

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score  # Import Silhouette Score

app = Flask(__name__)
CORS(app)

# Fungsi memproses file excel
def process_file(file_path):
    df = pd.read_excel(file_path)
    selected_columns = df.columns[3:]  # Ambil kolom dari D ke depan
    for column in selected_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    if df[selected_columns].isnull().any().any():
        return "Error: Beberapa nilai tidak valid (mungkin ada teks yang tidak bisa diubah menjadi angka)."
    return df[selected_columns]

# Fungsi menghitung Euclidean Distance
def euclidean_distance(point, centroids):
    return np.sqrt(np.sum((point - centroids) ** 2, axis=1))

# Inisialisasi centroid dengan K-Means++
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

# Fungsi K-Means
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

# Fungsi Elbow Method
def find_optimal_k(data, max_k=10):
    distortions = []
    for k in range(1, max_k + 1):
        centroids, clusters = k_means_clustering(data, k)
        intra_cluster_distances = np.min(cdist(data, centroids, 'euclidean'), axis=1)
        distortions.append(np.sum(intra_cluster_distances ** 2))
    return distortions

# Fungsi Elbow Method menggunakan sklearn
def find_optimal_k_sklearn(data, max_k=10):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)  # inertia_ adalah sum of squared distances (distorsi)
    
    # Visualisasi elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.grid(True)
    plt.show()

    return distortions

# API Endpoint
@app.route('/kmeans', methods=['POST'])
def kmeans():
    try:
        file = request.files['file']
        file_path = "temp_file.xlsx"
        file.save(file_path)
        df = process_file(file_path)
        if isinstance(df, str):
            return jsonify({'error': df}), 400

        data = df.values

        # distortions = find_optimal_k_sklearn(data)
        distortions = find_optimal_k(data)
        # optimal_k = distortions.index(min(distortions)) + 1
        optimal_k = 4

        centroids, clusters = k_means_clustering(data, optimal_k)
        df['Cluster'] = clusters
        
        # Hitung Silhouette Score
        silhouette_avg = silhouette_score(data, clusters)

        return jsonify({
            "optimal_k": optimal_k,
            "silhouette_score": silhouette_avg,
            "final_centroids": centroids.tolist(),
            "data": df.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
