from .utils.GDGT_compounds import *
from .utils.folder_handling import *
from .utils.messages import *
from .utils.handle_window_params import *
from .utils.import_data import *
from .utils.time_normalization import *
from .chromatoPy_base import *
from .utils.errors.smoothing_check import *

import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.interpolate as interp
import json


def hplc_integration(folder_path=None, windows=True, peak_neighborhood_n=5, smoothing_window=12, smoothing_factor=3, gaus_iterations=4000, maximum_peak_amplitude=None, peak_boundary_derivative_sensitivity=0.01, peak_prominence=0.001): # peak_boundary_derivative_sensitivity=0.01
    """
    Interactive integration of HPLC results. Steps to use.
    1. import the package
    2. Run the function "hplc_integration"
    3. Provide a filepath for the .csv output files from openchrom ("" or '' do not matter)
    4. Click peaks of interest. For traces with multiple peaks i.e., GDGT isomers, ensure that
        the 5-methyl (cren) is selected before the 6-meethyl (cren'). If the peak of interest is
        not available, click the position where the peak should be to set a blank peak holder.
        This is important for proper functinoality of chromatoPy. Peak-placement holders can be
        deleted by engaging the 'd' key.
    5. Advance to next GDGT group by engaging the 'enter' key, once all peaks are selected.
    6. Once a sample is complete, the results are saved to a .csv file in the a results folder
        within the user-provided filepath.

    Note: The code can be foribly stopped and finished samples will be saved. Upon calling the
    hplc_integration() funciton and providing the same filepath, the software will check the
    results folder, identify which samples were already processed, and continue with the next
    sample. To reproces|s a sample, simply delete it from the "results.csv" file

    Parameters
    ----------
    folder_path : String, optional
        Filepath string to the .csv files output from openChrom
    windows : Boolean, optional
        If True, chromatopy will use default windows values for window width (time, minute dimension) for figures.
        If False, the user will be prompted to provide window widths (time, minute dimension).
    peak_neighbrhood_n: Integer, optional
        Maximum number of peaks that will be considered a part of the peak neighborhood.
    gaus_it : Integer, optional
        Number of iterations to fit the (multi)gaussian curve. The default is 5000.

    Returns
    -------
    None.


    """
    # Error check smoothing values
    smooth           = smoothing_check(smoothing_window, smoothing_factor)
    smoothing_window = smooth['sw']
    smoothing_factor = smooth["sf"]
    
    # Display introduction message
    display_introduction_message()
    
    # Handle folder-related operations
    folder_info       = folder_handling(folder_path)
    folder_path       = folder_info["folder_path"]
    csv_files         = folder_info["csv_files"]
    sample_path       = folder_info['sample_path']
    output_folder     = folder_info["output_folder"]
    figures_folder    = folder_info["figures_folder"]
    results_file_path = folder_info["results_file_path"]
    # results_rts_path = folder_info["results_rts_path"]
    # results_area_unc_path = folder_info["results_area_unc_path"]
    ref_pk            = folder_info["ref_pk"]
    gdgt_oi           = folder_info["gdgt_oi"]
    gdgt_meta_set     = folder_info["gdgt_meta_set"]
    default_windows   = folder_info["default_windows"]
    gdgt_groups       = folder_info['names']
    
    # Handle window operations
    window_info = hand_window_params(windows, default_windows, gdgt_meta_set)
    windows     = window_info["windows"]
    GDGT_dict   = window_info["GDGT_dict"]
    trace_ids   = window_info["trace_ids"]
    
    # Handle data input
    data_info  = import_data(results_file_path, folder_path, csv_files, trace_ids) # results_rts_path, results_area_unc_path,
    data       = data_info["data"]
    reference  = data_info["reference"]
    results_df = data_info["results_df"]
    # results_rts_df = data_info["results_rts_df"]
    # result_area_unc_df = data_info["results_area_unc_df"]

    # Normalize time accross different samples
    time_norm = time_normalization(data)
    data      = time_norm["data"]
    iref      = time_norm["iref"]
       
    # Process samples
    for df in data:
        sample_name = df["Sample Name"].iloc[0]
        sample_file = {}
        sample_file['ID'] = sample_name
        if sample_name in results_df["Sample Name"].values:
            continue
        peak_data = {"Sample Name": sample_name}
        sample = {"Sample Name": sample_name}
        # time_data = {"Sample Name": sample_name}
        # peak_unc_data = {"Sample Name": sample_name}
        trace_sets = gdgt_meta_set["Trace"]
        trace_labels = gdgt_meta_set["names"]
        if iref:
            refpkhld = None
        else:
            refpkhld = ref_pk
        for trace_set, trace_label, window, GDGT_dict_single, gdgt_group in zip(trace_sets, trace_labels, windows, GDGT_dict, gdgt_groups):
            analyzer = GDGTAnalyzer(
                df, trace_set, window, GDGT_dict_single, gaus_iterations, sample_name, is_reference=iref, 
                max_peaks=peak_neighborhood_n, sw=smoothing_window, sf=smoothing_factor, max_PA = None,
                pk_sns=peak_boundary_derivative_sensitivity, pk_pr=peak_prominence, reference_peaks=refpkhld)
            # print(f"Begin peak selection for {sample_name}.")
            peaks, fig, ref_pk_new, t_pressed = analyzer.run()
            if iref:
                ref_pk.update(ref_pk_new)
            elif t_pressed:
                ref_pk.update(peaks)
                print(f"Reference peaks updated using {sample_name}.")
            all_gdgt_names = [item for sublist in GDGT_dict_single.values() for item in (sublist if isinstance(sublist, list) else [sublist])]
            # Iterate over all possible GDGTs
            sample[gdgt_group[0]] = {}
            for gdgt in all_gdgt_names:
                if gdgt in peaks:
                    peak_data[gdgt] = peaks[gdgt]["areas"][0]  # Assume there is only one area per compound
                    sample[gdgt_group[0]][gdgt] = peaks[gdgt]
                else:
                    peak_data[gdgt] = 0  # Use NaN if the GDGT is missing
                    sample[gdgt_group[0]][gdgt] = 0      
            fig_path = os.path.join(figures_folder, f"{sample_name}_{trace_label}.png")
            fig.savefig(fig_path)
            plt.close(fig)
        filename = sample['Sample Name'] + ".json"
        sample_out_path = os.path.join(sample_path, filename)
        os.makedirs(sample_path, exist_ok=True)
        with open(sample_out_path, "w", encoding="utf-8") as outfile:
            json.dump(sample, outfile, indent=3)
        new_entry = pd.DataFrame([peak_data])
        results_df = pd.concat([results_df, new_entry], ignore_index=True)
        results_df.to_csv(results_file_path, index=False)
        
        if not iref:
            refpkhld = ref_pk
        iref = False  # Only the first sample is treated as the reference
    
    print("Finished.")
