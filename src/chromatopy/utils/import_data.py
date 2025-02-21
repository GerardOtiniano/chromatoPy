# src/chromatopy/utils/import_data.py
import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def read_data_concurrently(folder_path, files, trace_ids):
    """
    Reads and cleans data from multiple files concurrently.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing the data files.
    files : list of str
        List of filenames to be read from the folder.
    trace_ids : list of str
        List of trace identifiers to filter the data by.

    Returns
    -------
    results : list of pandas.DataFrame
        List of dataframes containing cleaned data for each file.
    """
    def load_and_clean_data(file):
        full_path = os.path.join(folder_path, file)

        # Load the entire CSV (no usecols) to check for column name variations
        df = pd.read_csv(full_path)

        # Extracting sample name from filename and storing it in the DataFrame
        df["Sample Name"] = os.path.basename(file)[:-4]

        # Rename the RT column if "RT(minutes) - NOT USED BY IMPORT" is present
        if "RT(minutes) - NOT USED BY IMPORT" in df.columns:
            df.rename(columns={"RT(minutes) - NOT USED BY IMPORT": "RT (min)"}, inplace=True)

        # Remove ".0" from column names
        cleaned_columns = [col[:-2] if col.endswith(".0") else col for col in df.columns]
        df.columns = cleaned_columns

        # Ensure trace_ids are mapped correctly to columns that may have ended with ".0"
        for trace_id in trace_ids:
            if trace_id not in df.columns:
                # Check if trace_id with ".0" exists in columns
                trace_id_with_dot_zero = trace_id + ".0"
                if trace_id_with_dot_zero in df.columns:
                    # Rename the column with ".0" to match the trace_id
                    df.rename(columns={trace_id_with_dot_zero: trace_id}, inplace=True)

        # Filter DataFrame to only include the required columns (RT and Trace IDs)
        required_columns = ["Sample Name"] + ["RT (min)"] + trace_ids
        df = df[[col for col in df.columns if col in required_columns]]
        return df

    # Execute concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(load_and_clean_data, files))

    return results


def numerical_sort_key(filename):
    """
    Extracts numbers from the filename and returns them as an integer for sorting purposes.

    Parameters
    ----------
    filename : str
        The filename from which to extract the numerical values for sorting.

    Returns
    -------
    int
        The first numerical value found in the filename as an integer. If no numbers are found, returns 0.

    Notes
    -----
    - This function is typically used to sort files in numerical order based on the number(s) in their names.
    - If multiple numbers are present in the filename, only the first one is considered.
    """
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else 0

def import_data(results_file_path, folder_path, csv_files, trace_ids):
    """
    Imports and reads the chromatographic data and existing results for HPLC analysis.
    
    This function reads in previously processed results (if available) from the specified file path. It also reads and loads the chromatographic data from CSV files located in the provided folder path and processes the trace IDs.
    
    Parameters
    ----------
    results_file_path : str
        The file path to the CSV file containing previously saved results.
    folder_path : str
        The folder path where the input CSV files from openChrom are stored.
    csv_files : list
        List of CSV files to be processed, which contain chromatographic data.
    trace_ids : list
        List of trace identifiers used to extract relevant data from the CSV files.
    
    Returns
    -------
    dict
        A dictionary containing:
        - "data" (list): The processed chromatographic data for each sample.
        - "reference" (pandas.DataFrame): The first dataset, treated as the reference sample.
        - "results_df" (pandas.DataFrame): A dataframe containing previously processed results, if available, or an empty dataframe if none exist.
    
    Notes
    -----
    - The first dataset in the data list is treated as the reference sample and is stored in the "reference" key of the returned dictionary.
    - If the results file doesn't exist at the specified path, an empty dataframe is created and returned in the "results_df".
    - The data is read concurrently using the `read_data_concurrently` function for efficient data loading.
    """
    # get or read results path
    if os.path.exists(results_file_path):
        results_df = pd.read_csv(results_file_path)
        # results_rts_df = pd.read_csv(results_rts_path)
        # results_area_unc_df = pd.read_csv(results_area_unc_path)
    else:
        results_df = pd.DataFrame(columns=["Sample Name"])
        # results_rts_df = pd.DataFrame(columns=["Sample Name"])
        # results_area_unc_df = pd.DataFrame(columns=["Sample Name"])

    print("Reading data...")
    data = read_data_concurrently(folder_path, csv_files, trace_ids)
    reference = data[0]
        
    return {
        "data": data,
        "reference": reference,
        "results_df": results_df,}
        # "results_rts_df": results_rts_df,
        # "results_area_unc_df": results_area_unc_df}



