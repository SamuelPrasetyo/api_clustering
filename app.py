# Import Algoritma kmeans.py
from kmeans import k_means_clustering, evaluate_clustering_kmeans, find_optimal_k_sklearn

# Import Algoritma dbscan.py
from dbscan import DBSCAN, find_optimal_dbscan_params, evaluate_clustering_dbscan, plot_k_distance_graph

# Import Algoritma agglomerative.py
from agglomerative import agglomerative_clustering, evaluate_clustering_agglomerative, calculate_centroids, calculate_sse

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

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



@app.route('/elbow-method', methods=['POST'])
def elbow_method():
    try:
        # Ambil parameter dari request
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

        # Pastikan kolom metadata tersedia
        metadata_columns = ['Semester', 'Tahun Ajar', 'Kelas', 'NIS', 'Nama Siswa']
        clustering_columns = df.columns.difference(metadata_columns)

        # Pisahkan metadata dan data numerik untuk clustering
        clustering_data = df[clustering_columns]

        # Pastikan data numerik valid
        for column in clustering_columns:
            clustering_data.loc[:, column] = pd.to_numeric(clustering_data[column], errors='coerce')
        
        # Filter baris yang memiliki nilai null atau NaN
        valid_data_mask = ~clustering_data.isnull().any(axis=1)
        clustering_data = clustering_data[valid_data_mask]

        # Filter baris yang memiliki nilai `-`
        invalid_characters_mask = ~(clustering_data.apply(lambda row: any(str(x).strip() == '-' for x in row), axis=1))
        clustering_data = clustering_data[invalid_characters_mask]

        # Pastikan tidak ada data kosong setelah filtering
        if clustering_data.empty:
            return jsonify({'error': 'Semua data tidak valid untuk analisis.'}), 400
        
        # Jalankan Elbow Method
        data = clustering_data.values
        plot_url, distortions = find_optimal_k_sklearn(data)

        # Kembalikan hasil
        return jsonify({
            "plot": plot_url,
            "distortions": distortions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/kmeans', methods=['POST'])
def kmeans():
    try:
        # Ambil parameter tahun ajar dan semester dari request
        data = request.get_json()
        tahunajar = data.get('tahunajar')
        semester = data.get('semester')
        n_clusters = data.get('n_clusters')
        
        # Validasi parameter
        if not tahunajar or not semester or not n_clusters:
            return jsonify({'error': 'Parameter tahunajar, semester, dan n_clusters harus disediakan.'}), 400
        
         # Validasi n_clusters adalah bilangan positif
        try:
            n_clusters = int(n_clusters)
            if n_clusters <= 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'Parameter n_clusters harus berupa bilangan bulat positif.'}), 400
        
        # Filter data berdasarkan tahunajar dan semester
        df = get_data_from_db()
        df = df[(df['Tahun Ajar'] == tahunajar) & (df['Semester'] == semester)]
        
        if df.empty:
            return jsonify({'error': 'Tidak ada data untuk tahun ajar dan semester yang dipilih.'}), 400

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
        centroids, clusters = k_means_clustering(data, n_clusters)
        
        # Evaluasi hasil clustering
        evaluation = evaluate_clustering_kmeans(data, clusters, centroids)

        # Tambahkan hasil clustering ke DataFrame metadata
        metadata['Cluster'] = clusters

        # Gabungkan metadata dengan hasil clustering
        result_df = pd.concat([metadata, clustering_data], axis=1)
        
        # Sorting berdasarkan Nama Siswa
        result_df = result_df.sort_values(by=['Cluster', 'Kelas'])

        # Kembalikan hasil
        return jsonify({
            "data": result_df.to_dict(orient='records'),
            "final_centroids": centroids.tolist(),
            "n_clusters": n_clusters,
            "evaluation": evaluation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400



@app.route('/find-params', methods=['POST'])
def find_params():
    try:
        # Ambil data request
        data = request.get_json()
        eps_range = np.arange(data.get('eps_min', 7), data.get('eps_max', 10), data.get('eps_step', 0.5))
        min_pts_range = range(data.get('min_pts_min', 12), data.get('min_pts_max', 22), data.get('min_pts_step', 5))

        # Ambil data dari database
        df = get_data_from_db()
        tahunajar = data.get('tahunajar')
        semester = data.get('semester')
        df = df[(df['Tahun Ajar'] == tahunajar) & (df['Semester'] == semester)]

        # Metadata dan clustering columns
        metadata_columns = ['Semester', 'Tahun Ajar', 'Kelas', 'NIS', 'Nama Siswa']
        clustering_columns = df.columns.difference(metadata_columns)

        clustering_data = df[clustering_columns]
        for column in clustering_columns:
            clustering_data.loc[:, column] = pd.to_numeric(clustering_data[column], errors='coerce')

        clustering_data = clustering_data.dropna()
        data_array = clustering_data.values

        # Panggil fungsi plotting dari dbscan.py
        k_distance_plot = plot_k_distance_graph(data_array)
        
        # Temukan parameter optimal
        results = find_optimal_dbscan_params(data_array, eps_range, min_pts_range)

        return jsonify({
            'results': results, 
            'k_distance_plot': f'data:image/png;base64,{k_distance_plot}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dbscan', methods=['POST'])
def dbscan():
    try:
        # Ambil parameter dari request
        data = request.get_json()
        tahunajar = data.get('tahunajar')
        semester = data.get('semester')
        eps = data.get('eps')
        min_pts = data.get('min_pts')
        
        # Validasi parameter
        if not tahunajar or not semester or eps is None or min_pts is None:
            return jsonify({'error': 'Parameter tahunajar, semester, eps, dan min_pts harus disediakan.'}), 400
        
        # Validasi eps dan min_pts adalah angka valid
        try:
            eps = float(eps)
            min_pts = int(min_pts)
            if eps <= 0 or min_pts <= 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'Parameter eps harus berupa angka positif dan min_pts harus berupa bilangan bulat positif.'}), 400
        
        # Ambil data dari database
        df = get_data_from_db()
        df = df[(df['Tahun Ajar'] == tahunajar) & (df['Semester'] == semester)]
        
        if df.empty:
            return jsonify({'error': 'Tidak ada data untuk tahun ajar dan semester yang dipilih.'}), 400

        # Metadata dan clustering columns
        metadata_columns = ['Semester', 'Tahun Ajar', 'Kelas', 'NIS', 'Nama Siswa']
        clustering_columns = df.columns.difference(metadata_columns)

        # Pisahkan metadata dan data numerik
        metadata = df[metadata_columns]
        clustering_data = df[clustering_columns]

        # Validasi data numerik
        for column in clustering_columns:
            clustering_data.loc[:, column] = pd.to_numeric(clustering_data[column], errors='coerce')
        
        # Filter nilai NaN
        valid_data_mask = ~clustering_data.isnull().any(axis=1)
        clustering_data = clustering_data[valid_data_mask]
        metadata = metadata[valid_data_mask]

        # Filter nilai `-`
        invalid_characters_mask = ~(clustering_data.applymap(lambda x: str(x).strip() == '-').any(axis=1))
        clustering_data = clustering_data[invalid_characters_mask]
        metadata = metadata[invalid_characters_mask]

        # Pastikan tidak ada data kosong setelah filtering
        if clustering_data.empty:
            return jsonify({'error': 'Semua data tidak valid untuk clustering setelah memfilter nilai null atau -.'}), 400

        # Konversi data ke numpy array untuk DBSCAN
        data = clustering_data.values

        # Inisialisasi dan jalankan DBSCAN
        dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        labels = dbscan.fit(data)

        # Tambahkan hasil clustering ke metadata
        metadata['Cluster'] = labels

        # Evaluasi hasil clustering
        evaluation = evaluate_clustering_dbscan(data, labels)

        # Gabungkan metadata dan hasil clustering
        result_df = pd.concat([metadata, clustering_data], axis=1)
        result_df = result_df.sort_values(by=['Cluster', 'Kelas'])

        # Konversi hasil ke JSON
        response = {
            "evaluation": evaluation,
            # "sum_squared_error": sse,
            "data": result_df.to_dict(orient='records')
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400    


@app.route('/agglomerative', methods=['POST'])
def agglomerative():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Request body harus berupa JSON.'}), 400

        tahunajar = data.get('tahunajar')
        semester = data.get('semester')
        n_clusters = int(data.get('n_clusters', 3))  # Default to 3 clusters if not specified

        # Validasi parameter
        if not tahunajar or not semester:
            return jsonify({'error': 'Parameter tahunajar dan semester harus disediakan.'}), 400

        # Filter data berdasarkan tahunajar dan semester
        df = get_data_from_db()
        df = df[(df['Tahun Ajar'] == tahunajar) & (df['Semester'] == semester)]

        if df.empty:
            return jsonify({'error': 'Data tidak ditemukan untuk parameter yang diberikan.'}), 404

        # Pastikan kolom metadata tersedia
        metadata_columns = ['Semester', 'Tahun Ajar', 'Kelas', 'NIS', 'Nama Siswa']
        clustering_columns = df.columns.difference(metadata_columns)

        # Pisahkan metadata dan data numerik untuk clustering
        metadata = df[metadata_columns]
        clustering_data = df[clustering_columns]

        # Konversi data ke numerik
        clustering_data = clustering_data.apply(pd.to_numeric, errors='coerce')

        # Drop NaN dan invalid values
        clustering_data = clustering_data.dropna()
        metadata = metadata.loc[clustering_data.index]

        # Pastikan data tidak kosong setelah pembersihan
        if clustering_data.empty:
            return jsonify({'error': 'Data tidak valid untuk clustering setelah memfilter nilai null atau invalid.'}), 400

        # Jalankan Agglomerative Clustering
        clustering_data_values = clustering_data.values
        labels = agglomerative_clustering(clustering_data_values, n_clusters)

        # Evaluasi Clustering
        evaluation = evaluate_clustering_agglomerative(clustering_data, labels)
        
        # Tambahkan centroid setiap cluster untuk analisis
        centroids = {}
        unique_labels = np.unique(labels)
        for cluster_id in unique_labels:
            cluster_points = clustering_data[labels == cluster_id]
            centroid = cluster_points.mean(axis=0)
            centroids[cluster_id] = centroid

        # Tambahkan centroid ke hasil evaluasi untuk analisis lebih lanjut
        centroids = calculate_centroids(clustering_data, labels)

        # Hitung SSE
        sse = calculate_sse(clustering_data, labels, n_clusters)

        # Tambahkan hasil clustering ke metadata
        metadata['Cluster'] = labels

        # Gabungkan metadata dengan hasil clustering
        result_df = pd.concat([metadata, clustering_data], axis=1)

        # Sorting berdasarkan Nama Siswa
        result_df = result_df.sort_values(by=['Cluster', 'Kelas'])

        # Kembalikan hasil
        return jsonify({
            "data": result_df.to_dict(orient='records'),
            "centroids": centroids,
            "evaluation": {
                **evaluation,
                "sum_squared_error": sse
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500    
    


if __name__ == '__main__':
    app.run(debug=True)