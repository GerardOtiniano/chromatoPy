"""
# from FID_General import plot_chromatogram
# import os
# from PIL import Image
# import imagehash
# import numpy as np
# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# import matplotlib.pyplot as plt

# def clusterer(data, chromatogram_folder):
#     """
#     Function to cluster samples. Data contains the 

#     Parameters
#     ----------
#     data : TYPE
#         Dictinoary data structure produced using import_data().
#     chromatogram_folder : String
#         Pathway to folder containing chromatograms for clustering via perceptual hashing

#     Returns
#     -------
#     None.

#     """

#     # Get list of all image file paths
#     image_files = [f for f in os.listdir(chromatogram_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#     image_paths = [os.path.join(chromatogram_folder, f) for f in image_files]
#     sample_names = [os.path.splitext(f)[0] for f in image_files]
    
#     # Define function to compute perceptual hash
#     def get_perceptual_hash(image_path):
#         img = Image.open(image_path).convert("L")  # grayscale
#         return imagehash.phash(img)  # alternatives: dhash, ahash
    
#     # Compute hashes
#     hashes = [get_perceptual_hash(path) for path in image_paths]
    
#     # Compute pairwise hash distances (Hamming distances)
#     n = len(hashes)
#     dist_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i+1, n):
#             dist = hashes[i] - hashes[j]  # hamming distance
#             dist_matrix[i, j] = dist_matrix[j, i] = dist
    
#     # Perform hierarchical clustering
#     Z = linkage(dist_matrix, method='average')
    
#     # Plot dendrogram
#     plt.figure(figsize=(10, 6))
#     dendrogram(Z, labels=sample_names, leaf_rotation=90)
#     plt.tight_layout()
#     plt.show()
    
#     # Extract clusters (e.g., up to 5 groups)
#     clusters = fcluster(Z, t=5, criterion='maxclust')
    
#     # Optional: print assignments
#     for name, group in zip(sample_names, clusters):
#         print(f"{name}: Group {group}")
#         data['Samples'][name]['cluster'] = group
        

import os
from PIL import Image
import imagehash
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
# from FID_integration import import_data, FID_integration
# from FID_General import plot_chromatogram, plot_chromatogram_cluster

# def cluster(data, folder_path):
#     # data, no_time_col, no_signal_col, time_column, signal_column, folder_path = import_data()
#     clusterer(data,folder_path)
#     return data

def clusterer(data, chromatogram_folder, max_clusters=10, dendro_diagram=False):
    """
    Cluster chromatograms using perceptual hashing and determine optimal number
    of clusters via BIC on Gaussian Mixture Models.
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

    if dendro_diagram:
        Z = linkage(dist_matrix, method='average')
        plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=sample_names, leaf_rotation=90)
        plt.tight_layout()
        plt.show()

    # BIC plot
    plt.figure()
    plt.plot(range(2, max_clusters + 1), bic_scores, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("BIC")
    plt.show()

    # 7. Store cluster assignments in `data`
    for name, group in zip(sample_names, cluster_labels):
        # print(f"{name}: Group {group + 1}")
        data['Samples'][name]['cluster'] = int(group + 1)  # 1-based indexing
def get_cluster_labels(data):
    cluster_labels = set()
    for key in data['Samples'].keys():
        cluster_labels.add(data['Samples'][key]['cluster'])
    return cluster_labels
        
