import numpy as np
import pandas as pd
from scipy.signal import find_peaks, correlate
from scipy.interpolate import interp1d
import os
from concurrent.futures import ThreadPoolExecutor
import re


def distribute_peaks_to_gdgts(peaks, gdgt_list):
    """
    Distributes peak data to the corresponding GDGT compounds based on the provided list.

    Parameters
    ----------
    peaks : dict
        A dictionary containing peak data, where keys are compounds, and values are dictionaries with "areas" and "rts" (retention times).
    gdgt_list : list of str
        A list of GDGT compounds to map peaks to.

    Returns
    -------
    gdgt_peak_map : dict
        A dictionary mapping each GDGT to its corresponding peak data (area and retention time). If not enough peaks are found or too many are present, warnings are printed.
    """
    gdgt_peak_map = {gdgt: {"area": 0, "rt": None} for gdgt in gdgt_list}
    peak_items = []
    for gdgt, data in peaks.items():
        for area, rt in zip(data["areas"], data["rts"]):
            peak_items.append({"area": area, "rt": rt})
    for gdgt, peak in zip(gdgt_list, peak_items):
        if gdgt in gdgt_peak_map:
            gdgt_peak_map[gdgt] = {"area": peak["area"], "rt": peak["rt"]}
        else:
            print(f"Error: GDGT {gdgt} not found in map")  # Error handling
    if len(peak_items) < len(gdgt_list):
        print("Warning: Fewer peaks than expected. Check the output for correctness.")
    elif len(peak_items) > len(gdgt_list):
        print("Error: Too many peaks selected. Check the selections.")
    return gdgt_peak_map


def find_optimal_shift(reference, signal):
    """
    Finds the optimal shift between the reference and signal using cross-correlation.

    Parameters
    ----------
    reference : numpy.ndarray
        The reference signal (e.g., a chromatogram) to which the signal will be aligned.
    signal : numpy.ndarray
        The signal to be aligned to the reference.

    Returns
    -------
    lag : int
        The shift (lag) value that maximizes the correlation between the reference and signal.
    """
    correlation = correlate(reference, signal, mode="full", method="auto")
    lag = np.argmax(correlation) - (len(signal) - 1)
    return lag


def align_samples(data, trace_ids, reference):
    """
    Aligns sample data based on the reference signals using the optimal shift.

    Parameters
    ----------
    data : list of pandas.DataFrame
        List of dataframes containing chromatographic data for each sample.
    trace_ids : list of str
        List of trace identifiers used to align the data.
    reference : pandas.DataFrame
        The reference data used for alignment.

    Returns
    -------
    aligned_data : list of pandas.DataFrame
        List of aligned dataframes with corrected retention times based on the reference.
    """
    reference_signals = [reference[trace_id].dropna() for trace_id in trace_ids if trace_id in reference]
    if not reference_signals:
        print("No reference signals found. Time correction not applied.")
        return data  # Return original data if no valid reference signals found
    reference_composite = np.nanmean(np.array(reference_signals), axis=0)
    aligned_data = []
    for df in data:
        try:
            composite_signals = [df[trace_id].dropna() for trace_id in trace_ids if trace_id in df]
            if not composite_signals:
                aligned_data.append(df)
                continue  # Skip alignment if no signals are found for this sample
            composite = np.nanmean(np.array(composite_signals), axis=0)
            shift = find_optimal_shift(reference_composite, composite)
            df["rt_corr"] = df["RT(minutes) - NOT USED BY IMPORT"] - shift / 60  # Convert shift to minutes
            aligned_data.append(df)
        except Exception as e:
            print(f"Error processing {df}: {e}")
            aligned_data.append(df)  # Append unmodified DataFrame in case of an error
    return aligned_data


def discrete_time_shift(refy, lower, upper, name):
    """
    Applies a discrete time shift based on the specified upper and lower bounds for a given reference.

    Parameters
    ----------
    refy : pandas.DataFrame
        The reference dataframe containing the signal to be analyzed.
    lower : float
        The lower bound for the time shift.
    upper : float
        The upper bound for the time shift.
    name : str
        The column name to use for the time shift.

    Returns
    -------
    disc_time : pandas.Series
        The time-shifted reference signal within the specified bounds.
    """
    refy = refy.loc[(refy[name] < upper) & (refy[name] > lower)]
    refy = refy.reset_index(drop=True)
    pks, pks_meta = find_peaks(refy["744"], prominence=10, height=100)
    refy = refy.loc[refy["744"] == refy.loc[pks]["744"].max()]
    disc_time = refy[name]
    return disc_time


def interpolate_traces(df, trace_ids):
    """
    Interpolates all traces in the dataframe using cubic interpolation and updates the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing chromatographic data, including retention times and trace signals.
    trace_ids : list of str
        List of trace identifiers corresponding to the traces to be interpolated.

    Returns
    -------
    new_df : pandas.DataFrame
        The dataframe with interpolated traces and updated retention times.
    """
    x = df["RT(minutes) - NOT USED BY IMPORT"]
    x_new = np.linspace(x.min(), x.max(), num=len(x) * 4)  # Increase the number of x points
    new_df = pd.DataFrame(index=x_new)
    new_df["Sample Name"] = df["Sample Name"].iloc[0]  # Assuming it's consistent across the DataFrame
    for trace_id in trace_ids:
        y = df[trace_id]
        try:
            f = interp1d(x, y, kind="cubic", bounds_error=False, fill_value="extrapolate")
            y_new = f(x_new)
            new_df[trace_id] = y_new
        except ValueError as e:
            print(f"Error interpolating trace {trace_id}: {e}")
    new_df["RT(minutes) - NOT USED BY IMPORT"] = x_new
    return new_df.reset_index(drop=True)


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
