#!/chromatoPy/src/chromatopy/FID/import_data.py
import os
from tqdm import tqdm
import re
import pandas as pd
import json

def import_data(): # folder_path=None):
    folder_path = input("Provide folder containing .txt files: ")
    folder_path = folder_path.strip('\'"')

    # List all .txt files
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".txt")]
    if not txt_files:
        tqdm.write(f"No .txt files found in {folder_path}. Aborting.")
        raise SystemExit

    output_path, figures_path = create_output_folders(folder_path)

    no_time_col = []
    no_signal_col = []
    data_dict = {}
    data_dict["Samples"] = {}

    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Find the line where the table begins
        table_start = None
        for i, line in enumerate(lines):
            if re.search(r'(?i)^\s*Chromatogram Data\s*:', line):
                table_start = i + 1  # The actual table header is the next line
                break

        if table_start is None or table_start + 1 >= len(lines):
            tqdm.write(f"Could not find table start in {filename}")
            print(f"Could not find table start in {filename}")
            continue

        # Read headers
        headers = lines[table_start].strip().split('\t')
        data_lines = lines[table_start + 1:]

        # Read into DataFrame
        try:
            df = pd.DataFrame([l.strip().split('\t') for l in data_lines if l.strip() != ''], columns=headers)
        except Exception as e:
            tqdm.write(f"Failed to parse table in {filename}: {e}")
            print(f"Failed to parse table in {filename}: {e}")
            continue

        # Header matching
        time_keywords = ['time', 'min', 'sec', 'second', 'minute']
        signal_keywords = ['signal', 'value', 'intensity', 'amplitude', '(pa)', '(a)']
        headers = lines[table_start].strip().split('\t')
        header_map = {h.lower(): h for h in headers}

        time_column = next((header_map[h] for h in header_map if any(key in h for key in time_keywords)), None)
        has_time = time_column is not None
        signal_column = next((header_map[h] for h in header_map if any(key in h for key in signal_keywords)), None)
        has_signal = signal_column is not None

        # Numeric dataframe
        df[time_column] = pd.to_numeric(df[time_column], errors='coerce')
        df[signal_column] = pd.to_numeric(df[signal_column], errors='coerce')
        df[signal_column] = df[signal_column]#.fillna(0)

        if not has_time:
            no_time_col.append(filename)
        if not has_signal:
            no_signal_col.append(filename)
        metadata = ''.join(lines[:table_start - 1])
        parsed_metadata = parse_metadata_block(metadata)

        # Store in dictionary
        data_dict['Samples'][filename.replace(".txt", "")] = {

            "Metadata": parsed_metadata,
            "Raw Data": {time_column:df[time_column].to_list(), signal_column:df[signal_column].to_list()}}#df.to_dict(orient="list")}
    tqdm.write(f"Found {len(txt_files)} .txt files.")
    if no_time_col:
        tqdm.write("Files missing time column:", no_time_col)
    if no_signal_col:
        tqdm.write("Files missing signal column:", no_signal_col)

    # Check against any existing dataset
    data_dict, unprocessed_keys = check_existing_jsons(data_dict, output_path)
    return {
        "data_dict": data_dict,
        "no_time_col": no_time_col,
        "no_signal_col": no_signal_col,
        "time_column": time_column,
        "signal_column": signal_column,
        "folder_path": folder_path,
        "output_path": output_path,
        "figures_path": figures_path,
        "unprocessed_samples": unprocessed_keys}

def parse_metadata_block(raw_text):
    """
    Parse chromatogram metadata string into a nested dictionary.
    """
    lines = raw_text.strip().split("\n")
    result = {}
    current_section = None

    for line in lines:
        if not line.strip():
            continue  # skip empty lines

        parts = line.split("\t")
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == 1:
            # This is likely a section header like "Injection Information:"
            section = parts[0].rstrip(":")
            result[section] = {}
            current_section = section
        elif len(parts) == 2:
            key, value = parts
            if current_section:
                result[current_section][key] = value
            else:
                result[key] = value
        else:
            # Unhandled line structure
            tqdm.write("Skipping malformed line:", line)
            print("Skipping malformed line:", line())

    # return result

def create_output_folders(folder_path):
    """
    Creates a 'chromatoPy output' folder inside the given folder_path.
    If it already exists, deletes it and recreates it.
    Also creates a nested 'Figures' subfolder.

    Returns
    -------
    output_path : str
        Path to 'chromatoPy output' folder.
    figures_path : str
        Path to 'chromatoPy output/Figures' folder.
    """
    output_path = os.path.join(folder_path, "chromatoPy output")
    figures_path = os.path.join(output_path, "Figures")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    return output_path, figures_path

def check_existing_jsons(data_dict, output_path):
    """
    Merge any new samples in data_dict into an existing JSON on disk.
    Returns: (merged_dict, unprocessed_keys)
      - merged_dict: either the existing JSON plus new samples, or your original data_dict
      - unprocessed_keys: list of sample-names that came in fresh
    """
    existing = load_json(output_path)
    # no file on disk → nothing processed yet
    if existing is None:
        unprocessed = list(data_dict["Samples"].keys())
        return data_dict, unprocessed

    # else: there was an existing JSON
    before = set(existing["Samples"].keys())
    # add any brand-new samples
    for name, sample in data_dict["Samples"].items():
        if name not in existing["Samples"]:
            existing["Samples"][name] = sample

    # figure out which ones we just added
    after = set(existing["Samples"].keys())
    unprocessed = list(after - before)

    return existing, unprocessed

def load_json(output_path, list_samples=False, list_processed=False):
    """
    Try to load FID_output.json from output_path.
    If it doesn’t exist, return None.
    Otherwise return the dict, rebuilding any Raw Data dicts into DataFrames.
    """
    js_file = os.path.join(output_path, "FID_output.json")
    if not os.path.exists(js_file):
        return None

    with open(js_file, "r") as f:
        data = json.load(f)

    # rebuild Raw Data dicts into DataFrames
    for sample in data.get("Samples", {}).values():
        raw = sample.get("Raw Data")
        if isinstance(raw, dict):
            sample["Raw Data"] = pd.DataFrame(raw)
    if list_samples:
        for key in data['Samples'].keys():
            print(key)
    if list_processed:
        key = []
        for x in data['Samples'].keys():
            if 'Processed Data' in data['Samples'][x].keys():
                key.append(x)
        print(key)
    return data

