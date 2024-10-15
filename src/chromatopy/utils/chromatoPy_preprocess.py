import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


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
