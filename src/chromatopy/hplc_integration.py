from .GDGT_compounds import *
from .chromatoPy_preprocess import *
from .chromatoPy_base import *

# from indices import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.interpolate as interp


def hplc_integration(folder_path=None, windows=True, peak_neighborhood_n=3, smoothing_window=12, smoothing_factor=3, gaus_iterations=4000, peak_boundary_derivative_sensitivity=0.05, peak_prominence=1):
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
    print(
        "Welcome to chromatopy. When prompted, provide a filepath to your data (.csv output from openchrom). Select the peaks of interest from the first sample. Chromatopy will then automatically select peaks in subsequent samples. After each sample, results and figures are saved to a subfolder created in the user-provided data directory. Chromatopy will export a .csv file containing peak areas. To calculate relative abundances and common indices, run chromatopy.assign_indices(). \nFeel free to end an integration session by terminating the kernel as your resutls will not be deleted. You can revisit samples at any time. To redo a sample, delete the row containing the integration data from the results.csv output.\n1. Left click peaks for integration.\n2. 'd' to delete the last peak selected.\n3. 'r' to clear selected peaks from a subplot (navigate subplots using up and down arrow keys).\n 4. 'Enter' once peak selection is satisfied.\n"
    )
    # Request folder location
    if folder_path == None:
        folder_path = input("Input folder location of converted .csv UHLPC files: ")
    folder_path = folder_path.replace('"', "")  # Remove quotations from filepath
    folder_path = folder_path.replace("'", "")
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")], key=numerical_sort_key)  # Find gdgt signal files (csv) from openChrom output
    # Set names of output folders and files
    output_folder = os.path.join(folder_path, "Output_chromatoPy")
    figures_folder = os.path.join(output_folder, "Figures_chromatoPy")
    results_file_path = os.path.join(output_folder, "results_peak_area.csv")
    os.makedirs(figures_folder, exist_ok=True)  # make figure subfolder
    ref_pk = {}
    # Ask for GDGTs of interest
    gdgt_oi = get_gdgt_input()
    gdgt_meta_set = get_gdgt(gdgt_oi)  # get metadata for GDGT types
    # Extract default windows
    default_windows = gdgt_meta_set["window"]

    # Handle custom windows
    if windows:
        # Use default windows
        windows = default_windows
    elif windows is False:
        # Prompt user to input custom windows
        windows = []
        print("\nYou have chosen to provide custom time windows for each GDGT group.")
        print("Please provide the time windows (in minutes) for each GDGT group.")
        print("The number of windows should match the number of GDGT groups selected.")
        print("For reference, the default windows are:")
        for idx, (gdgt_group, default_window) in enumerate(zip(gdgt_meta_set["names"], default_windows)):
            print(f"{idx + 1}. {gdgt_group}: {default_window}")
            # Prompt user for new window
            user_input = input(f"Enter new window for {gdgt_group} as two numbers separated by a comma (e.g., 10.5,12.0): ")
            try:
                lower, upper = map(float, user_input.split(","))
                windows.append([lower, upper])
            except ValueError:
                print("Invalid input. Please enter two numbers separated by a comma.")
                # You might want to handle retries or set default
                windows.append(default_window)  # Use default if invalid
    else:
        # Validate the provided windows
        if len(windows) != len(gdgt_meta_set["names"]):
            raise ValueError("The number of custom windows provided does not match the number of GDGT groups selected.")
    # windows = gdgt_meta_set["window"]
    GDGT_dict = gdgt_meta_set["GDGT_dict"]
    trace_ids = [x for trace in gdgt_meta_set["Trace"] for x in trace]
    # get or read results path
    if os.path.exists(results_file_path):
        results_df = pd.read_csv(results_file_path)
    else:
        results_df = pd.DataFrame(columns=["Sample Name"])

    print("Reading data...")
    data = read_data_concurrently(folder_path, csv_files, trace_ids)

    reference = data[0]
    # Normalize time accross different samples
    for d in data:
        time_change = discrete_time_shift(d, lower=10, upper=60, name="RT (min)")  # "RT(minutes) - NOT USED BY IMPORT")
        d["rt_corr"] = d["RT (min)"] - time_change.iloc[0] + 20  # "RT(minutes) - NOT USED BY IMPORT"] - time_change.iloc[0] + 20
    iref = True  # Flag to indicate the first sample (reference sample)
    if os.path.exists(results_file_path):
        results_df = pd.read_csv(results_file_path)
    else:
        results_df = pd.DataFrame(columns=["Sample Name"])
    for df in data:
        sample_name = df["Sample Name"].iloc[0]
        if sample_name in results_df["Sample Name"].values:
            continue
        peak_data = {"Sample Name": sample_name}
        trace_sets = gdgt_meta_set["Trace"]
        trace_labels = gdgt_meta_set["names"]
        if iref:
            refpkhld = None
        else:
            refpkhld = ref_pk
        for trace_set, trace_label, window, GDGT_dict_single in zip(trace_sets, trace_labels, windows, GDGT_dict):
            df2 = df.loc[(df["rt_corr"] > window[0]) & (df["rt_corr"] < window[1])]
            df2 = df2.reset_index(drop=True)
            analyzer = GDGTAnalyzer(
                df2, trace_set, window, GDGT_dict_single, gaus_iterations, sample_name, is_reference=iref, max_peaks=peak_neighborhood_n, sw=smoothing_window, sf=smoothing_factor, pk_sns=peak_boundary_derivative_sensitivity, pk_pr=peak_prominence, reference_peaks=refpkhld
            )  # Set parameters for HPLC analysis
            print("Begin peak selection.")
            peaks, fig, ref_pk_new = analyzer.run()
            if iref:
                ref_pk.update(ref_pk_new)
            all_gdgt_names = [item for sublist in GDGT_dict_single.values() for item in (sublist if isinstance(sublist, list) else [sublist])]

            # Iterate over all possible GDGTs
            for gdgt in all_gdgt_names:
                if gdgt in peaks:
                    peak_data[gdgt] = peaks[gdgt]["areas"][0]  # Assume there is only one area per compound
                else:
                    peak_data[gdgt] = 0  # Use NaN if the GDGT is missing

            fig_path = os.path.join(figures_folder, f"{sample_name}_{trace_label}.png")
            fig.savefig(fig_path)
            plt.close(fig)

        new_entry = pd.DataFrame([peak_data])
        results_df = pd.concat([results_df, new_entry], ignore_index=True)
        results_df.to_csv(results_file_path, index=False)
        if not iref:
            refpkhld = ref_pk
        iref = False  # Only the first sample is treated as the reference
    print("Finished.")
