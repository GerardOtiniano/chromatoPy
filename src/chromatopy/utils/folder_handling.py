# src/chromatopy/utils/folder_handling.py

import os
import pandas as pd

from .GDGT_configuration import load_gdgt_window_data
from .import_data import numerical_sort_key
from .GDGT_compounds import *

def folder_handling(folder_path):
    """
    Handles folder-related operations: input processing, CSV retrieval, directory setup, and GDGT selection.
    
    Parameters:
        folder_path (str or None): The path to the folder containing CSV files. If None, prompts user input.
        
    Returns:
        dict: A dictionary containing all necessary variables for further processing.
    """
    if folder_path is None:
        folder_path = input("Input folder location of converted .csv UHLPC files: ")
    
    # Clean the folder path by removing quotes
    folder_path = folder_path.replace('"', "").replace("'", "")
    
    # Retrieve and sort CSV files
    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".csv")],
        key=numerical_sort_key  # Ensure numerical_sort_key is defined or imported
    )
    
    # Define output directories and results file path
    output_folder = os.path.join(folder_path, "Output_chromatoPy")
    figures_folder = os.path.join(output_folder, "Figures_chromatoPy")
    results_file_path = os.path.join(output_folder, "results_peak_area.csv")
    sample_path = os.path.join(output_folder, "Individual Samples")
    # results_rts_path = os.path.join(output_folder, "results_rts.csv")
    # results_area_unc_path = os.path.join(output_folder, "results_area_uncertainty.csv")
    # Create figures folder if it doesn't exist
    os.makedirs(figures_folder, exist_ok=True)
    
    # Initialize reference peaks
    ref_pk = {}

    # Loads GDGT Data from the GDGT Settings
    gdgt_meta_set = load_gdgt_window_data()
    
    # Extract needed information
    default_windows = gdgt_meta_set["window"]
    names = gdgt_meta_set["names"]
    
    return {
        "folder_path": folder_path,
        "csv_files": csv_files,
        "output_folder": output_folder,
        "sample_path": sample_path,
        "figures_folder": figures_folder,
        "results_file_path": results_file_path,
        # "results_rts_path": results_rts_path,
        # "results_area_unc_path": results_area_unc_path,
        "ref_pk": ref_pk,
        "gdgt_meta_set": gdgt_meta_set,
        "default_windows": default_windows,
        "names": names
    }