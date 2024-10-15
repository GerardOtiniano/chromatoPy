# src/chromatopy/utils/time_normalization.py
import numpy as np
from scipy.signal import correlate, find_peaks

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


def time_normalization(data):
    """
    Normalizes the retention time (RT) across different samples by applying a discrete time shift and adjusting RT values accordingly.
    
    Parameters
    ----------
    data : list of pandas.DataFrame
        A list of dataframes where each dataframe represents a sample and contains a "RT (min)" column for retention times.
    
    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "data" (list of pandas.DataFrame): The modified list of dataframes with the normalized "rt_corr" column added.
        - "iref" (bool): A flag set to True to indicate that the first sample is treated as the reference sample.
    
    Notes
    -----
    - The function assumes that each dataframe in `data` contains a "RT (min)" column.
    - The `discrete_time_shift` function is used to compute a time shift within a lower and upper bound of 10 and 60 minutes, respectively.
    - The retention times in each dataframe are corrected based on this computed time shift, and the corrected values are stored in the "rt_corr" column.
    """
    for d in data:
        time_change = discrete_time_shift(d, lower=10, upper=60, name="RT (min)")  # "RT(minutes) - NOT USED BY IMPORT")
        d["rt_corr"] = d["RT (min)"] - time_change.iloc[0] + 20  # "RT(minutes) - NOT USED BY IMPORT"] - time_change.iloc[0] + 20
    iref = True  # Flag to indicate the first sample (reference sample)
    return {
        "data": data,
        "iref": iref}