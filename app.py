# Import Algoritma kmeans.py
from kmeans import k_means_clustering, evaluate_clustering_kmeans, find_optimal_k_sklearn

# Import Algoritma dbscan.py
from dbscan import DBSCAN, find_optimal_dbscan_params, evaluate_clustering_dbscan, plot_k_distance_graph, calculate_centroids

# Import Algoritma agglomerative.py
from agglomerative import agglomerative_clustering, evaluate_clustering_agglomerative, calculate_centroids, calculate_sse

# Import Library Evaluasi
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

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
        
        # Sorting berdasarkan Cluster dan Kelas
        result_df = result_df.sort_values(by=['Cluster', 'Kelas'])

        # Tambahkan nama mata pelajaran
        subjects = ['AGAMA', 'PKN', 'BAHASA INDONESIA', 'MATEMATIKA', 'IPA', 
                    'IPS', 'BAHASA INGGRIS', 'SENI BUDAYA', 'PJOK', 
                    'PRAKARYA', 'TIK']
        final_centroids_with_subjects = []
        for centroid in centroids:
            subject_centroid = {subject: value for subject, value in zip(subjects, centroid)}
            final_centroids_with_subjects.append(subject_centroid)

        # Kembalikan hasil
        return jsonify({
            "data": result_df.to_dict(orient='records'),
            # "final_centroids": centroids.tolist(),
            "final_centroids": final_centroids_with_subjects,
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
        eps_range = np.arange(data.get('eps_min', 6), data.get('eps_max', 15), data.get('eps_step', 0.5))
        # eps_range = np.arange(data.get('eps_min', 4), data.get('eps_max', 12), data.get('eps_step', 0.5))
        min_pts_range = range(data.get('min_pts_min', 12), data.get('min_pts_max', 22), data.get('min_pts_step', 1))

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
        
        # Hitung centroid setiap kluster
        centroids = calculate_centroids(data, labels)
        
        # Tambahkan nama mata pelajaran ke centroid
        subjects = ['AGAMA', 'PKN', 'BAHASA INDONESIA', 'MATEMATIKA', 'IPA', 
                    'IPS', 'BAHASA INGGRIS', 'SENI BUDAYA', 'PJOK', 
                    'PRAKARYA', 'TIK']

        centroids_with_subjects = {}
        for label, centroid_values in centroids.items():
            centroids_with_subjects[label] = {
                subject: value for subject, value in zip(subjects, centroid_values)
            }

        # Evaluasi hasil clustering
        evaluation = evaluate_clustering_dbscan(data, labels)

        # Gabungkan metadata dan hasil clustering
        result_df = pd.concat([metadata, clustering_data], axis=1)
        result_df = result_df.sort_values(by=['Cluster', 'Kelas'])

        # Konversi hasil ke JSON
        response = {
            "evaluation": evaluation,
            "centroids": centroids_with_subjects,  # Konversi centroids ke JSON-friendly format
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
        
        # Tambahkan hasil clustering ke metadata
        metadata['Cluster'] = labels
        
        # Hitung centroid setiap cluster
        centroids = calculate_centroids(clustering_data, labels)
        
        subjects = ['AGAMA', 'PKN', 'BAHASA INDONESIA', 'MATEMATIKA', 'IPA', 
                    'IPS', 'BAHASA INGGRIS', 'SENI BUDAYA', 'PJOK', 
                    'PRAKARYA', 'TIK']

        # Tambahkan nama mata pelajaran ke centroids
        centroids_with_subjects = {}
        for cluster_id, centroid_values in centroids.items():
            if isinstance(centroid_values, (list, np.ndarray)):
                # Pastikan centroid_values berupa list atau array
                centroids_with_subjects[cluster_id] = {
                    subject: value for subject, value in zip(subjects, centroid_values)
                }
            else:
                return jsonify({'error': f'Unexpected centroid structure for cluster {cluster_id}'}), 500

        # Evaluasi Clustering
        evaluation = evaluate_clustering_agglomerative(clustering_data, labels)

        # Gabungkan metadata dengan hasil clustering
        result_df = pd.concat([metadata, clustering_data], axis=1)

        # Sorting berdasarkan Nama Siswa
        result_df = result_df.sort_values(by=['Cluster', 'Kelas'])

        # Kembalikan hasil
        return jsonify({
            "data": result_df.to_dict(orient='records'),
            "centroids": centroids_with_subjects,
            "evaluation": {
                **evaluation
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500    
    
# @app.route('/hasil-perbandingan', methods=['POST'])
# def hasil_perbandingan():
    try:
        # Ambil data dari database
        df = get_data_from_db()
        
        if df.empty:
            return jsonify({'error': 'Tidak ada data untuk tahun ajar dan semester yang dipilih.'}), 400

        # Variasi Tahun Ajar dan Semester
        tahun_ajar_list = ['20212022', '20222023', '20232024']
        semester_list = ['Gasal', 'Genap']

        # Metadata dan clustering columns
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

        results = []

        # Loop melalui setiap kombinasi Tahun Ajar dan Semester
        for tahun_ajar in tahun_ajar_list:
            for semester in semester_list:
                subset = df[(df['Tahun Ajar'] == tahun_ajar) & (df['Semester'] == semester)]

                if subset.empty:
                    continue

                clustering_subset = subset[clustering_columns]
                # clustering_subset = subset[clustering_columns].copy()
                for column in clustering_columns:
                    # clustering_subset[column] = pd.to_numeric(clustering_subset[column], errors='coerce')
                    clustering_subset.loc[:, column] = pd.to_numeric(clustering_subset[column], errors='coerce')

                clustering_subset = clustering_subset.dropna()
                subset_array = clustering_subset.values

                # K-Means
                kmeans = k_means_clustering(subset_array, k=3)  # Asumsikan k=3
                kmeans_labels = kmeans['labels']
                kmeans_dbi = davies_bouldin_score(subset_array, kmeans_labels)
                kmeans_chi = calinski_harabasz_score(subset_array, kmeans_labels)
                kmeans_silhouette = silhouette_score(subset_array, kmeans_labels)

                # DBSCAN
                dbscan_results = find_optimal_dbscan_params(subset_array, np.arange(0.5, 2.5, 0.5), range(5, 10))
                best_dbscan = dbscan_results[0]  # Asumsikan hasil terbaik adalah yang pertama
                dbscan = DBSCAN(eps=best_dbscan['eps'], min_samples=best_dbscan['min_pts']).fit(subset_array)
                dbscan_labels = dbscan.labels_
                dbscan_dbi = davies_bouldin_score(subset_array, dbscan_labels)
                dbscan_chi = calinski_harabasz_score(subset_array, dbscan_labels)
                dbscan_silhouette = silhouette_score(subset_array, dbscan_labels)

                # Agglomerative
                agglomerative = agglomerative_clustering(subset_array, n_clusters=3)  # Asumsikan 3 klaster
                agglomerative_labels = agglomerative['labels']
                agglomerative_dbi = davies_bouldin_score(subset_array, agglomerative_labels)
                agglomerative_chi = calinski_harabasz_score(subset_array, agglomerative_labels)
                agglomerative_silhouette = silhouette_score(subset_array, agglomerative_labels)

                # Simpan hasil evaluasi
                results.append({
                    'tahun_ajar': tahun_ajar,
                    'semester': semester,
                    'kmeans': {
                        'davies_bouldin_index': kmeans_dbi,
                        'calinski_harabasz_index': kmeans_chi,
                        'silhouette_score': kmeans_silhouette
                    },
                    'dbscan': {
                        'davies_bouldin_index': dbscan_dbi,
                        'calinski_harabasz_index': dbscan_chi,
                        'silhouette_score': dbscan_silhouette
                    },
                    'agglomerative': {
                        'davies_bouldin_index': agglomerative_dbi,
                        'calinski_harabasz_index': agglomerative_chi,
                        'silhouette_score': agglomerative_silhouette
                    }
                })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/hasil-perbandingan-agglomerative', methods=['POST'])
def hasil_perbandingan_agglomerative():
    try:
        # Ambil data dari database
        df = get_data_from_db()

        if df.empty:
            return jsonify({'error': 'Tidak ada data untuk tahun ajar dan semester yang dipilih.'}), 400

        # Variasi Tahun Ajar dan Semester
        tahun_ajar_list = ['20212022', '20222023', '20232024']
        semester_list = ['Gasal', 'Genap']

        # Metadata dan clustering columns
        metadata_columns = ['Semester', 'Tahun Ajar', 'Kelas', 'NIS', 'Nama Siswa']
        clustering_columns = df.columns.difference(metadata_columns)

        # Filter data valid untuk clustering
        clustering_data = df[clustering_columns].copy()
        for column in clustering_columns:
            clustering_data[column] = pd.to_numeric(clustering_data[column], errors='coerce')

        clustering_data = clustering_data.dropna()  # Hapus baris dengan nilai NaN
        if clustering_data.empty:
            return jsonify({'error': 'Semua data tidak valid untuk clustering setelah memfilter nilai null atau -.'}), 400

        results = []

        # Loop melalui setiap kombinasi Tahun Ajar dan Semester
        for tahun_ajar in tahun_ajar_list:
            for semester in semester_list:
                subset = df[(df['Tahun Ajar'] == tahun_ajar) & (df['Semester'] == semester)]
                if subset.empty:
                    continue

                clustering_subset = subset[clustering_columns].copy()
                for column in clustering_columns:
                    clustering_subset[column] = pd.to_numeric(clustering_subset[column], errors='coerce')

                clustering_subset = clustering_subset.dropna()
                subset_array = clustering_subset.values

                # Agglomerative
                agglomerative = agglomerative_clustering(subset_array, n_clusters=3)  # Asumsikan 3 klaster
                if isinstance(agglomerative, dict):
                    agglomerative_labels = agglomerative.get('labels', None)
                elif isinstance(agglomerative, tuple):
                    agglomerative_labels = agglomerative[0]
                else:
                    agglomerative_labels = agglomerative

                # Validasi bahwa agglomerative_labels adalah array
                if agglomerative_labels is None or not isinstance(agglomerative_labels, (list, np.ndarray)):
                    return jsonify({'error': 'Output dari agglomerative_clustering tidak valid.'}), 400

                agglomerative_labels = np.array(agglomerative_labels).flatten()  # Pastikan 1D

                # Evaluasi Agglomerative
                agglomerative_dbi = davies_bouldin_score(subset_array, agglomerative_labels)
                agglomerative_chi = calinski_harabasz_score(subset_array, agglomerative_labels)
                agglomerative_silhouette = silhouette_score(subset_array, agglomerative_labels)

                # Simpan hasil evaluasi
                results.append({
                    'tahun_ajar': tahun_ajar,
                    'semester': semester,
                    'agglomerative': {
                        'davies_bouldin_index': agglomerative_dbi,
                        'calinski_harabasz_index': agglomerative_chi,
                        'silhouette_score': agglomerative_silhouette
                    }
                })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/hasil-perbandingan-k-means', methods=['POST'])
def hasil_perbandingan_k_means():
    try:
        # Ambil data dari database
        df = get_data_from_db()

        if df.empty:
            return jsonify({'error': 'Tidak ada data untuk tahun ajar dan semester yang dipilih.'}), 400

        # Metadata dan clustering columns
        metadata_columns = ['Semester', 'Tahun Ajar', 'Kelas', 'NIS', 'Nama Siswa']
        clustering_columns = df.columns.difference(metadata_columns)

        # Pisahkan metadata dan data numerik untuk clustering
        metadata = df[metadata_columns]
        clustering_data = df[clustering_columns]

        # Pastikan data numerik valid
        for column in clustering_columns:
            clustering_data[column] = pd.to_numeric(clustering_data[column], errors='coerce')

        # Filter baris yang memiliki nilai null atau NaN
        valid_data_mask = ~clustering_data.isnull().any(axis=1)
        clustering_data = clustering_data[valid_data_mask]
        metadata = metadata[valid_data_mask]

        # Filter baris yang memiliki nilai `-`
        invalid_characters_mask = ~(clustering_data.applymap(lambda x: str(x).strip() == '-').any(axis=1))
        clustering_data = clustering_data[invalid_characters_mask]
        metadata = metadata[invalid_characters_mask]

        # Pastikan tidak ada data kosong setelah filtering
        if clustering_data.empty:
            return jsonify({'error': 'Semua data tidak valid untuk clustering setelah memfilter nilai null atau -.'}), 400

        results = []

        # Loop melalui setiap kombinasi Tahun Ajar dan Semester
        for tahun_ajar in df['Tahun Ajar'].unique():
            for semester in df['Semester'].unique():
                subset = df[(df['Tahun Ajar'] == tahun_ajar) & (df['Semester'] == semester)]
                if subset.empty:
                    continue

                clustering_subset = subset[clustering_columns].copy()
                for column in clustering_columns:
                    clustering_subset[column] = pd.to_numeric(clustering_subset[column], errors='coerce')

                clustering_subset = clustering_subset.dropna()
                subset_array = clustering_subset.values

                # Periksa apakah data valid untuk clustering
                if subset_array.shape[0] == 0:
                    continue

                # K-Means
                kmeans = k_means_clustering(subset_array, k=3)
                if isinstance(kmeans, tuple):
                    _, kmeans_labels = kmeans
                else:
                    return jsonify({'error': 'Output K-Means tidak valid.'}), 400

                kmeans_labels = np.array(kmeans_labels).flatten()

                # Pastikan jumlah label sesuai
                if len(kmeans_labels) != subset_array.shape[0]:
                    return jsonify({
                        'panjang data': subset_array.shape[0],
                        'panjang label': len(kmeans_labels),
                        'error': f'Jumlah label ({len(kmeans_labels)}) tidak sesuai dengan jumlah data ({subset_array.shape[0]}).'
                    }), 400

                # Evaluasi K-Means
                kmeans_dbi = davies_bouldin_score(subset_array, kmeans_labels)
                kmeans_chi = calinski_harabasz_score(subset_array, kmeans_labels)
                kmeans_silhouette = silhouette_score(subset_array, kmeans_labels)

                # Simpan hasil evaluasi
                results.append({
                    'tahun_ajar': tahun_ajar,
                    'semester': semester,
                    'kmeans': {
                        'davies_bouldin_index': kmeans_dbi,
                        'calinski_harabasz_index': kmeans_chi,
                        'silhouette_score': kmeans_silhouette
                    }
                })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/hasil-perbandingan-dbscan', methods=['POST'])
def hasil_perbandingan_dbscan():
    try:
        # Ambil data dari database
        df = get_data_from_db()

        if df.empty:
            return jsonify({'error': 'Tidak ada data untuk tahun ajar dan semester yang dipilih.'}), 400

        # Metadata dan clustering columns
        metadata_columns = ['Semester', 'Tahun Ajar', 'Kelas', 'NIS', 'Nama Siswa']
        clustering_columns = df.columns.difference(metadata_columns)

        # Pisahkan metadata dan data numerik untuk clustering
        metadata = df[metadata_columns]
        clustering_data = df[clustering_columns]

        # Pastikan data numerik valid
        for column in clustering_columns:
            clustering_data[column] = pd.to_numeric(clustering_data[column], errors='coerce')

        # Filter baris yang memiliki nilai null atau NaN
        valid_data_mask = ~clustering_data.isnull().any(axis=1)
        clustering_data = clustering_data[valid_data_mask]
        metadata = metadata[valid_data_mask]

        # Filter baris yang memiliki nilai `-`
        invalid_characters_mask = ~(clustering_data.applymap(lambda x: str(x).strip() == '-').any(axis=1))
        clustering_data = clustering_data[invalid_characters_mask]
        metadata = metadata[invalid_characters_mask]

        # Pastikan tidak ada data kosong setelah filtering
        if clustering_data.empty:
            return jsonify({'error': 'Semua data tidak valid untuk clustering setelah memfilter nilai null atau -.'}), 400

        results = []

        # Loop melalui setiap kombinasi Tahun Ajar dan Semester
        for tahun_ajar in df['Tahun Ajar'].unique():
            for semester in df['Semester'].unique():
                subset = df[(df['Tahun Ajar'] == tahun_ajar) & (df['Semester'] == semester)]
                if subset.empty:
                    continue

                clustering_subset = subset[clustering_columns].copy()
                for column in clustering_columns:
                    clustering_subset[column] = pd.to_numeric(clustering_subset[column], errors='coerce')

                clustering_subset = clustering_subset.dropna()
                subset_array = clustering_subset.values

                # Periksa apakah data valid untuk clustering
                if subset_array.shape[0] == 0:
                    continue

                # Cari parameter terbaik untuk DBSCAN
                eps_range = np.arange(6, 15, 0.5) # Variasi eps
                min_pts_range = range(12, 22, 1) # Variasi min_pts

                optimal_params = find_optimal_dbscan_params(subset_array, eps_range, min_pts_range)

                if len(optimal_params) == 0:
                    continue

                best_param = optimal_params[0]  # Ambil parameter terbaik
                dbscan_model = DBSCAN(eps=best_param['eps'], min_pts=best_param['min_pts'])
                labels = dbscan_model.fit(subset_array)

                # Evaluasi hasil clustering
                evaluation = evaluate_clustering_dbscan(subset_array, labels)

                # Simpan hasil evaluasi
                results.append({
                    'tahun_ajar': tahun_ajar,
                    'semester': semester,
                    'optimal_params': {
                        'eps': best_param['eps'],
                        'min_pts': best_param['min_pts']
                    },
                    'evaluation': evaluation
                })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/perbandingan-total-poin', methods=['POST'])
def perbandingan_total_poin():
    try:
        # Daftar tahun ajar dan semester yang akan diproses
        tahun_ajar_list = ['20212022', '20222023', '20232024']
        semester_list = ['Gasal', 'Genap']

        # Fungsi untuk mengambil hasil evaluasi dari setiap algoritma
        def get_results_from_algorithm(algorithm_route):
            response = app.test_client().post(algorithm_route, json={})
            if response.status_code == 200:
                return response.json.get('results', [])
            else:
                return []

        # Ambil hasil evaluasi dari setiap algoritma
        kmeans_results = get_results_from_algorithm('/hasil-perbandingan-k-means')
        dbscan_results = get_results_from_algorithm('/hasil-perbandingan-dbscan')
        agglomerative_results = get_results_from_algorithm('/hasil-perbandingan-agglomerative')

        # Total poin untuk setiap algoritma di seluruh kombinasi
        total_points = {'kmeans': 0, 'dbscan': 0, 'agglomerative': 0}

        # Loop melalui semua kombinasi tahun ajar dan semester
        for tahun_ajar in tahun_ajar_list:
            for semester in semester_list:
                # Filter hasil untuk kombinasi tahun ajar dan semester tertentu
                def filter_results_by_tahun_ajar_semester(results, tahun_ajar, semester):
                    return next((result for result in results if result['tahun_ajar'] == tahun_ajar and result['semester'] == semester), None)

                kmeans_result = filter_results_by_tahun_ajar_semester(kmeans_results, tahun_ajar, semester)
                dbscan_result = filter_results_by_tahun_ajar_semester(dbscan_results, tahun_ajar, semester)
                agglomerative_result = filter_results_by_tahun_ajar_semester(agglomerative_results, tahun_ajar, semester)

                if not kmeans_result or not dbscan_result or not agglomerative_result:
                    continue  # Jika salah satu hasil tidak ditemukan, lewati kombinasi ini

                # Perbandingan skor evaluasi untuk kombinasi ini
                metrics = ['davies_bouldin_index', 'calinski_harabasz_index', 'silhouette_score']
                algorithms = ['kmeans', 'dbscan', 'agglomerative']

                # Bandingkan setiap metrik untuk kombinasi saat ini
                for metric in metrics:
                    scores = {
                        'kmeans': kmeans_result['kmeans'][metric],
                        'dbscan': dbscan_result['evaluation'][metric],
                        'agglomerative': agglomerative_result['agglomerative'][metric],
                    }

                    # Cari algoritma dengan skor terbaik
                    if metric == 'davies_bouldin_index':  # DBI, lebih kecil lebih baik
                        best_algorithm = min(scores, key=scores.get)
                    else:  # Silhouette Score dan CHI, lebih besar lebih baik
                        best_algorithm = max(scores, key=scores.get)

                    # Beri poin ke algoritma terbaik
                    total_points[best_algorithm] += 1

        return jsonify({'total_points': total_points})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/perbandingan-total-poin2', methods=['POST'])
def perbandingan_total_poin2():
    try:
        # Daftar tahun ajar dan semester yang akan diproses
        tahun_ajar_list = ['20212022', '20222023', '20232024']
        semester_list = ['Gasal', 'Genap']

        # Fungsi untuk mengambil hasil evaluasi dari setiap algoritma
        def get_results_from_algorithm(algorithm_route):
            response = app.test_client().post(algorithm_route, json={})
            if response.status_code == 200:
                return response.json.get('results', [])
            else:
                return []

        # Ambil hasil evaluasi dari setiap algoritma
        kmeans_results = get_results_from_algorithm('/hasil-perbandingan-k-means')
        dbscan_results = get_results_from_algorithm('/hasil-perbandingan-dbscan')
        agglomerative_results = get_results_from_algorithm('/hasil-perbandingan-agglomerative')

        # Total skor untuk setiap algoritma
        total_scores = {
            'kmeans': {'davies_bouldin_index': 0, 'calinski_harabasz_index': 0, 'silhouette_score': 0},
            'dbscan': {'davies_bouldin_index': 0, 'calinski_harabasz_index': 0, 'silhouette_score': 0},
            'agglomerative': {'davies_bouldin_index': 0, 'calinski_harabasz_index': 0, 'silhouette_score': 0}
        }

        # Loop melalui semua kombinasi tahun ajar dan semester
        for tahun_ajar in tahun_ajar_list:
            for semester in semester_list:
                # Filter hasil untuk kombinasi tahun ajar dan semester tertentu
                def filter_results_by_tahun_ajar_semester(results, tahun_ajar, semester):
                    return next((result for result in results if result['tahun_ajar'] == tahun_ajar and result['semester'] == semester), None)

                kmeans_result = filter_results_by_tahun_ajar_semester(kmeans_results, tahun_ajar, semester)
                dbscan_result = filter_results_by_tahun_ajar_semester(dbscan_results, tahun_ajar, semester)
                agglomerative_result = filter_results_by_tahun_ajar_semester(agglomerative_results, tahun_ajar, semester)

                if not kmeans_result or not dbscan_result or not agglomerative_result:
                    continue  # Jika salah satu hasil tidak ditemukan, lewati kombinasi ini

                # Tambahkan skor evaluasi ke total
                total_scores['kmeans']['davies_bouldin_index'] += kmeans_result['kmeans']['davies_bouldin_index']
                total_scores['kmeans']['calinski_harabasz_index'] += kmeans_result['kmeans']['calinski_harabasz_index']
                total_scores['kmeans']['silhouette_score'] += kmeans_result['kmeans']['silhouette_score']

                total_scores['dbscan']['davies_bouldin_index'] += dbscan_result['evaluation']['davies_bouldin_index']
                total_scores['dbscan']['calinski_harabasz_index'] += dbscan_result['evaluation']['calinski_harabasz_index']
                total_scores['dbscan']['silhouette_score'] += dbscan_result['evaluation']['silhouette_score']

                total_scores['agglomerative']['davies_bouldin_index'] += agglomerative_result['agglomerative']['davies_bouldin_index']
                total_scores['agglomerative']['calinski_harabasz_index'] += agglomerative_result['agglomerative']['calinski_harabasz_index']
                total_scores['agglomerative']['silhouette_score'] += agglomerative_result['agglomerative']['silhouette_score']

        # Hitung total skor keseluruhan
        overall_scores = {alg: sum(scores.values()) for alg, scores in total_scores.items()}

        return jsonify({'total_scores': total_scores, 'overall_scores': overall_scores})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)