# src/chromatopy/utils/folder_handling.py

import os
import pandas as pd
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
    
    # Create figures folder if it doesn't exist
    os.makedirs(figures_folder, exist_ok=True)
    
    # Initialize reference peaks
    ref_pk = {}
    
    # Prompt user for GDGTs of interest
    gdgt_oi = get_gdgt_input()  # Ensure get_gdgt_input is defined or imported
    gdgt_meta_set = get_gdgt(gdgt_oi)  # Ensure get_gdgt is defined or imported
    
    # Extract default windows
    default_windows = gdgt_meta_set["window"]
    
    return {
        "folder_path": folder_path,
        "csv_files": csv_files,
        "output_folder": output_folder,
        "figures_folder": figures_folder,
        "results_file_path": results_file_path,
        "ref_pk": ref_pk,
        "gdgt_oi": gdgt_oi,
        "gdgt_meta_set": gdgt_meta_set,
        "default_windows": default_windows
    }