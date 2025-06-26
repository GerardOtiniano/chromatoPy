import os
import shutil
from PIL import Image
import imagehash
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from .FID_integration import import_data, save_json
from .FID_General import plot_chromatogram, plot_chromatogram_cluster

def cluster(time_window=None, max_clusters = 4, copy_files = True):
    data, chrom_folder , output_path, folder_path = plot_chromatogram(time_window)
    data = clusterer(data, chrom_folder, max_clusters=max_clusters)
    if copy_files:
        cluster_files(data, folder_path)
    save_json({"data_dict": data}, output_path)

def clusterer(data, chromatogram_folder, max_clusters):#, dendro_diagram=False):
    """
    Cluster chromatograms using perceptual hashing and determine optimal number
    of clusters via BIC on Gaussian Mixture Models.

    Parameters
    ----------
    data : dict
        Dictinoary data structure produced using import_data().
    chromatogram_folder : str
        Pathway to folder containing chromatograms for clustering via perceptual hashing.
    max_clusters : int
        Maximum number of clusters to split data into. Default 4.
    dendro_diagram: bool
        Figure displaying the number of clusters and assocaited BIC. Defualt False.

    Returns
    -------
    None.

    """
    # Load chromatogram images
    image_files = [f for f in os.listdir(chromatogram_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_paths = [os.path.join(chromatogram_folder, f) for f in image_files]
    sample_names = [os.path.splitext(f)[0] for f in image_files]
    
    def get_perceptual_hash(image_path):
        img = Image.open(image_path).convert("L")
        return imagehash.phash(img)
    
    hashes = [get_perceptual_hash(path) for path in image_paths]

    # Compute Hamming distance matrix
    n = len(hashes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = hashes[i] - hashes[j]
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Dimensionality reduction using MDS
    mds = MDS(n_components=4, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(dist_matrix)

    # BIC-based selection of optimal number of clusters
    lowest_bic = np.infty
    best_gmm = None
    bic_scores = []

    for k in range(2, max_clusters + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(embedding)
        bic = gmm.bic(embedding)
        bic_scores.append(bic)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
    print(f"Optimal number of clusters identified by BIC: {best_gmm.n_components}")

    optimal_clusters = best_gmm.n_components
    cluster_labels = best_gmm.predict(embedding)

    # BIC plot
    plt.figure()
    plt.plot(range(2, max_clusters + 1), bic_scores, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("BIC")
    plt.show()

    for name, group in zip(sample_names, cluster_labels):
        data['Samples'][name]['cluster'] = int(group + 1)  # 1-based indexing
    return data

def cluster_files(data, folder_path):
    assignments = {
        name: sample_info["cluster"]
        for name, sample_info in data["Samples"].items()}
    
    for sample_name, cluster_num in assignments.items():
        src = os.path.join(folder_path, f"{sample_name}.txt")
        dest_dir = os.path.join(folder_path, f"Cluster {cluster_num}")
        os.makedirs(dest_dir, exist_ok=True)
        try:
            shutil.copy(src, dest_dir)
        except FileNotFoundError:
            print(f"⚠️  raw file not found: {src}")
        # else:
        #     print(f"→ Copied {sample_name}.txt to Cluster {cluster_num}/")