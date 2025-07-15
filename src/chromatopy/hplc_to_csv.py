# src/chromatopy/hplc_to_csv.py
import os
import pandas as pd
import rainbow as rb


def hplc_to_csv(base_path=None):
    """
    Processes each .D folder within the base_path, reads any .MS file,
    creates a DataFrame, and saves it as a CSV file in the specified output folder.

    Parameters:
    - base_path: str, path to the directory containing .D folders
    """
    # Ensure the output directory exists
    if base_path is None:
        base_path = input("Enter parent folder pathway to raw HPLC data: ")
        
    print()
    output_base_path = os.path.join(base_path, "chromatopy - raw hplc csv")
    os.makedirs(output_base_path, exist_ok=True)
    # Iterate over all items in base_path assuming they are directories ending with .D
    for folder in os.listdir(base_path):
        if folder.endswith(".D"):
            folder_path = os.path.join(base_path, folder)
            datadir = rb.read(folder_path)
            for file in datadir.datafiles:
                if ".MS" in file.name:
                    output_name = os.path.join(output_base_path, str(folder).replace(".D", ".csv"))
                    out = os.path.join(output_base_path, output_name)
                    datadir.export_csv(str(file), out)
