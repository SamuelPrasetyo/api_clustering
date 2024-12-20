from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

app = Flask(__name__)
CORS(app)

# Konfigurasi Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/tugasakhir_2111501157'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Model Database
class NilaiSiswa(db.Model):
    __tablename__ = 'nilaisiswa'
    idnilai = db.Column(db.Integer, primary_key=True)  # Primary key
    semester = db.Column(db.String(15), nullable=False)
    tahunajar = db.Column(db.String(8), nullable=False)
    nis = db.Column(db.String(11), nullable=False)
    kelas = db.Column(db.String(10), nullable=False)
    nama_siswa = db.Column(db.String(255), nullable=False)
    nagama = db.Column(db.String(3), nullable=True)
    npkn = db.Column(db.String(3), nullable=True)
    nbindo = db.Column(db.String(3), nullable=True)
    nmatematika = db.Column(db.String(3), nullable=True)
    nipa = db.Column(db.String(3), nullable=True)
    nips = db.Column(db.String(3), nullable=True)
    nbinggris = db.Column(db.String(3), nullable=True)
    nsenibudaya = db.Column(db.String(3), nullable=True)
    npjok = db.Column(db.String(3), nullable=True)
    nprakarya = db.Column(db.String(3), nullable=True)
    ntik = db.Column(db.String(3), nullable=True)

# Fungsi memproses data dari database
def get_data_from_db():
    query = db.session.query(NilaiSiswa).all()
    data = [
        {
            "Semester": item.semester,
            "Tahun Ajar": item.tahunajar,
            "Kelas": item.kelas,
            "NIS": item.nis,
            "Nama Siswa": item.nama_siswa,
            "NAGAMA": item.nagama,
            "NPKN": item.npkn,
            "NBINDO": item.nbindo,
            "NMATEMATIKA": item.nmatematika,
            "NIPA": item.nipa,
            "NIPS": item.nips,
            "NBINGGRIS": item.nbinggris,
            "NSENIBUDAYA": item.nsenibudaya,
            "NPJOK": item.npjok,
            "NPRAKARYA": item.nprakarya,
            "NTIK": item.ntik,
        }
        for item in query
    ]
    df = pd.DataFrame(data)
    return df



# Fungsi memproses file excel
def process_file(file_path):
    df = pd.read_excel(file_path)
    selected_columns = df.columns[5:]  # Ambil kolom dari F dan seterusnya
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

@app.route('/kmeans', methods=['POST'])
def kmeans():
    try:
        # Ambil parameter tahun ajar dan semester dari request
        data = request.get_json()
        tahunajar = data.get('tahunajar')
        semester = data.get('semester')
        
        # Validasi parameter
        if not tahunajar or not semester:
            return jsonify({'error': 'Parameter tahunajar dan semester harus disediakan.'}), 400
        
        # Filter data berdasarkan tahunajar dan semester
        df = get_data_from_db()
        df = df[(df['Tahun Ajar'] == tahunajar) & (df['Semester'] == semester)]
        
        if df.empty:
            return jsonify({'error': 'Tidak ada data untuk tahun ajar dan semester yang dipilih.'}), 400
        
        # Cetak nama kolom untuk debug
        print("Nama kolom dalam DataFrame:", df.columns.tolist())

        # Pastikan kolom metadata tersedia
        metadata_columns = ['Semester', 'Tahun Ajar', 'Kelas', 'NIS', 'Nama Siswa']
        clustering_columns = df.columns.difference(metadata_columns)

        # Pisahkan metadata dan data numerik untuk clustering
        metadata = df[metadata_columns]
        clustering_data = df[clustering_columns]

        # Pastikan data numerik valid
        for column in clustering_columns:
            clustering_data[column] = pd.to_numeric(clustering_data[column], errors='coerce')
        
        # Filter baris yang memiliki nilai null atau NaN
        valid_data_mask = ~clustering_data.isnull().any(axis=1)  # True untuk baris tanpa NaN
        clustering_data = clustering_data[valid_data_mask]
        metadata = metadata[valid_data_mask]

        # Filter baris yang memiliki nilai `-`
        invalid_characters_mask = ~(clustering_data.applymap(lambda x: str(x).strip() == '-').any(axis=1))
        clustering_data = clustering_data[invalid_characters_mask]
        metadata = metadata[invalid_characters_mask]

        # Pastikan tidak ada data kosong setelah filtering
        if clustering_data.empty:
            return jsonify({'error': 'Semua data tidak valid untuk clustering setelah memfilter nilai null atau -.'}), 400
        
        # Jalankan K-Means
        data = clustering_data.values
        optimal_k = 3  # Atau gunakan metode untuk menentukan optimal_k
        centroids, clusters = k_means_clustering(data, optimal_k)

        # Tambahkan hasil clustering ke DataFrame metadata
        metadata['Cluster'] = clusters

        # Evaluasi hasil clustering
        silhouette_avg = silhouette_score(data, clusters)
        davies_bouldin = davies_bouldin_score(data, clusters)
        calinski_harabasz = calinski_harabasz_score(data, clusters)
        
        # Hitung Sum of Squared Error (SSE)
        sse = np.sum(np.min(cdist(data, centroids, 'euclidean') ** 2, axis=1))

        # Gabungkan metadata dengan hasil clustering
        result_df = pd.concat([metadata, clustering_data], axis=1)
        
        # Sorting berdasarkan Nama Siswa
        result_df = result_df.sort_values(by=['Cluster', 'Kelas'])

        # Kembalikan hasil
        return jsonify({
            "data": result_df.to_dict(orient='records'),
            "final_centroids": centroids.tolist(),
            "optimal_k": optimal_k,
            "silhouette_score": silhouette_avg,
            "davies_bouldin_index": davies_bouldin,
            "calinski_harabasz_index": calinski_harabasz,
            "sum_squared_error": sse
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)
    
# @app.route('/kmeans', methods=['POST'])
# def kmeans():
#     try:
#         # Ambil file yang dikirim dari request
#         file = request.files['file']

#         # Simpan file sementara di server
#         file_path = "temp_file.xlsx"
#         file.save(file_path)

#         # Proses file
#         df = pd.read_excel(file_path)

#         # Pastikan kolom metadata tersedia
#         metadata_columns = ['Semester', 'Tahun Ajar', 'Kelas', 'NIS', 'Nama Siswa']
#         clustering_columns = df.columns.difference(metadata_columns)

#         # Pisahkan metadata dan data numerik untuk clustering
#         metadata = df[metadata_columns]
#         clustering_data = df[clustering_columns]

#         # Pastikan data numerik valid
#         for column in clustering_columns:
#             clustering_data[column] = pd.to_numeric(clustering_data[column], errors='coerce')
        
#         if clustering_data.isnull().any().any():
#             return jsonify({'error': 'Beberapa nilai tidak valid dalam data clustering.'}), 400
        
#         # Jalankan K-Means
#         data = clustering_data.values
#         optimal_k = 3  # Atau gunakan metode untuk menentukan optimal_k
#         centroids, clusters = k_means_clustering(data, optimal_k)

#         # Tambahkan hasil clustering ke DataFrame metadata
#         metadata['Cluster'] = clusters

#         # Evaluasi hasil clustering
#         silhouette_avg = silhouette_score(data, clusters)
#         davies_bouldin = davies_bouldin_score(data, clusters)
#         calinski_harabasz = calinski_harabasz_score(data, clusters)
        
#         # Hitung Sum of Squared Error (SSE)
#         sse = np.sum(np.min(cdist(data, centroids, 'euclidean') ** 2, axis=1))

#         # Gabungkan metadata dengan hasil clustering
#         result_df = pd.concat([metadata, clustering_data], axis=1)
        
#         # Sorting berdasarkan Nama Siswa
#         result_df = result_df.sort_values(by=['Kelas', 'Semester', 'Cluster'])

#         # Kembalikan hasil
#         return jsonify({
#             "data": result_df.to_dict(orient='records'),
#             "final_centroids": centroids.tolist(),
#             "optimal_k": optimal_k,
#             "silhouette_score": silhouette_avg,
#             "davies_bouldin_index": davies_bouldin,
#             "calinski_harabasz_index": calinski_harabasz,
#             "sum_squared_error": sse
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400
    
# if __name__ == '__main__':
#     app.run(debug=True)