import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import hdbscan
from tqdm import tqdm

from .FID_Integration_functions import *
from .import_data import import_data

import os
import shutil


def extract_peak_times_from_sample(sample_df, time_range=(2, 20), prominence=1):
    time = sample_df['Time (min)'].values
    signal = sample_df['Value (pA)'].values
    mask = (time >= time_range[0]) & (time <= time_range[1])
    if not np.any(mask):
        return np.array([])

    time_in_range = time[mask]
    signal_in_range = signal[mask]
    norm_signal = signal_in_range / np.max(signal_in_range)
    _, min_peak_amp = baseline(time_in_range, norm_signal)
    peaks, _ = find_peaks(norm_signal, prominence=prominence, height = min_peak_amp)
    return time_in_range[peaks]

def peak_time_distance_with_threshold(a, b, threshold):
    if len(a) == 0 or len(b) == 0:
        return 1.0  # maximum dissimilarity

    def count_unmatched_and_error(x, y):
        matched = []
        unmatched = 0
        total_error = 0
        for xi in x:
            diffs = np.abs(y - xi)
            if np.any(diffs <= threshold):
                closest = np.min(diffs)
                total_error += closest
                matched.append(xi)
            else:
                unmatched += 1
        return unmatched, total_error

    unmatched_a, error_a = count_unmatched_and_error(a, b)
    unmatched_b, error_b = count_unmatched_and_error(b, a)

    total_peaks = len(a) + len(b)
    total_unmatched = unmatched_a + unmatched_b
    total_error = error_a + error_b

    # Normalize to get a distance between 0 and 1
    normalized_distance = (total_unmatched + total_error / threshold) / total_peaks
    return normalized_distance

def compute_timing_distance_matrix(peak_time_list, threshold=0.2):
    n = len(peak_time_list)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = peak_time_distance_with_threshold(peak_time_list[i], peak_time_list[j], threshold=threshold)
            dist[i, j] = dist[j, i] = d
    return dist


def cluster_samples_by_peak_timing(data_dict, min_cluster_size, min_samples, time_range=(2, 20), prominence=10):
    sample_names = list(data_dict['Samples'].keys())
    peak_times_list = []

    for name in tqdm(sample_names, desc="Processing chromatography similairty"):
        df = data_dict['Samples'][name]['raw data']

        time_col = next((col for col in df.columns if 'Time (min)' in col), None)
        signal_col = next((col for col in df.columns if 'Value (pA)' in col), None)

        if 'Time (min)' in df.columns and 'Value (pA)' in df.columns:
            peak_times = extract_peak_times_from_sample(df[['Time (min)', 'Value (pA)']], time_range, prominence)
            peak_times_list.append(peak_times)
        else:
            tqdm.write(f"Skipping {name}: missing 'Time (min)' or 'Value (pA)' columns: {df.columns}")
            peak_times_list.append(np.array([]))

    dist_matrix = compute_timing_distance_matrix(peak_times_list)
    clusterer = hdbscan.HDBSCAN(metric='precomputed',min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(dist_matrix)
    cluster_labels = np.unique(labels)

    # Add cluster label to each sample's dictionary
    for name, label in zip(sample_names, labels):
        data_dict['Samples'][name]['cluster'] = int(label)
    return cluster_labels

def move_files_to_cluster_subfolder(file_names, cluster_var, source_folder):
    # Create destination subfolder
    dest_folder = os.path.join(source_folder, f"cluster_{cluster_var}")
    os.makedirs(dest_folder, exist_ok=True)

    # Normalize file names (remove extension for matching)
    for filename in os.listdir(source_folder):
        full_path = os.path.join(source_folder, filename)
        if os.path.isfile(full_path):
            name_without_ext = os.path.splitext(filename)[0]
            if name_without_ext in file_names:
                shutil.move(full_path, os.path.join(dest_folder, filename))
                tqdm.write(f"Moved {filename} → cluster_{cluster_var}/")

    tqdm.write(f"All matching files moved to {dest_folder}.")
    
def cluster_figures(data):
    cluster_labels = set()
    for key in data['Samples'].keys():
        cluster_labels.add(data['Samples'][key]['cluster'])
    for cl in cluster_labels:
        fig=plt.figure()
        for key in data['Samples'].keys():
            if data['Samples'][key]['cluster'] != cl: continue
            plt.plot(data['Samples'][key]['raw data']['Time (min)'], data['Samples'][key]['raw data']['Value (pA)'], c= 'k')
        plt.xlabel("Time (min)")
        plt.ylabel(f"Value (pA)\n{cl}")
        plt.show()
        
def get_cluster_labels(data):
    cluster_labels = set()
    for key in data['Samples'].keys():
        cluster_labels.add(data['Samples'][key]['cluster'])
    return cluster_labels

def plot_chromatogram(time_window=None):
    """
    Plot and save chromatograms of all FID samples.

    Parameters
    ----------
    time_window : list of two floats, optional
        [xmin, xmax] time limits (in same units as input) to restrict the plot.
        If None, the full trace is plotted.
    """
    # 1) load data & get the raw folder path
    result = import_data()
    data = result['data_dict']
    time_column = result["time_column"]
    signal_column = result["signal_column"]
    folder_path = result["folder_path"]
    output_path = result["output_path"]
    figures_path = result["figures_path"]
    
    # 2) make (or reuse) the output folder
    chromatograms_dir = os.path.join(folder_path, "chromatograms")
    os.makedirs(chromatograms_dir, exist_ok=True)

    # 3) loop and save each sample’s trace
    for key, sample in data['Samples'].items():
        df = pd.DataFrame({
            'x': sample['Raw Data'][time_column],
            'y': sample['Raw Data'][signal_column]
        })
        if time_window is not None:
            xmin, xmax = time_window
            df = df.loc[(df.x >= xmin) & (df.x <= xmax)]

        plt.figure()
        plt.plot(df.x, df.y, color='k')
        plt.xlabel(time_column)
        plt.ylabel(signal_column)

        out_file = os.path.join(chromatograms_dir, f"{key}.png")
        plt.savefig(out_file, dpi=300)
        plt.close()
    return data, chromatograms_dir, output_path, folder_path
        
def plot_chromatogram_cluster(data, time_window=[4,30]):
    """
    Function for plotting the categorized chromatograms.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    time_window : TYPE, optional
        DESCRIPTION. The default is [4,30].

    Returns
    -------
    None.

    """
    cluster_n = 0
    for key in data['Samples'].keys():
        if 'cluster' not in data['Samples'][key].keys():
            tqdm.write("Data does not appear to contain cluster labels. Please run clusterer().")
            return
        else:
            if data['Samples'][key]['cluster'] > cluster_n:
                cluster_n = data['Samples'][key]['cluster']
    for i in range(1,cluster_n+1):
        fig = plt.figure()
        count = 0
        for key in data['Samples'].keys():
            # if group_assignments[key] ==val:
            if data['Samples'][key]['cluster']==i:
                count +=1
                temp = data['Samples'][key]['Raw Data']
                temp = temp[temp['Time (min)']>4]
                plt.plot(temp['Time (min)'], temp[f'Value (pA)'], c='k', alpha = 0.4)
        plt.text(x=0.05, y = 0.95, s=f"n = {count}", transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top')
        plt.ylabel(f"Signal (pA)\nCluster {i}")
        plt.xlabel("Time (min)")
        plt.show()