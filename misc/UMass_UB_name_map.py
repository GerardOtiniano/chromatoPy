#!/usr/bin/env python3
import os
import pandas as pd
import sys

"""
Script to create .csv containing paired UMass and UB lab names.

Instructions 
 - Put .py file in main raw data folder. 
 From terminal/command line, run:
"python UMass_UB_name_map.py /path/to/main/folder"

Output
- A .csv file with paried UMass and UB sample names saved in the folder
containing raw HPLC data.
"""

def extract_data_from_file(file_path, print_name=False):
    lab_name = None
    sample_name = None
    with open(file_path, "r", encoding="utf-16") as file:
        for line in file:
            if line.startswith("Sample Name: "):
                sample_name = line.strip().split("Sample Name: ")[1]
    return sample_name

def collect_data_from_directories(root_dir):
    data = []  # This will store tuples of (subfolder_name, lab_name, sample_name)
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "Report.TXT":
                file_path = os.path.join(subdir, file)
                sample_name = extract_data_from_file(file_path)
                subfolder_name = os.path.basename(subdir)  # Get the subfolder name
                UMass_name = subfolder_name.replace(".D", "")
                if sample_name:
                    data.append((subfolder_name, UMass_name, sample_name))
    return data

def create_dataframe(data, folder_path):
    df = pd.DataFrame(data, columns=["Subfolder Name", "UMass Lab ID", "UB Lab ID"])
    df.to_csv(os.path.join(folder_path,"UMass_UB_ID.csv"), index=False)

def map_HPLC_names(folder_path):
    data = collect_data_from_directories(folder_path)
    create_dataframe(data, folder_path)
    
if __name__ == '__main__':
    # If a folder path is provided as a command-line argument, use it.
    # Otherwise, default to the current working directory.
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = os.getcwd()

    map_HPLC_names(folder_path)