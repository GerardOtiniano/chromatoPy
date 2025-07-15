import os
from chromatopy import chromatopy_general
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import re

def read_data_concurrently(folder_path, files):
    def load_and_clean_data(file):
        full_path = os.path.join(folder_path, file)
        df = pd.read_csv(full_path)

        df["Sample Name"] = os.path.basename(file)[:-4]

        if "RT(minutes) - NOT USED BY IMPORT" in df.columns:
            df.rename(columns={"RT(minutes) - NOT USED BY IMPORT": "RT (min)"}, inplace=True)

        df.columns = [col[:-2] if col.endswith(".0") else col for col in df.columns]

        required_columns = ["Sample Name", "RT (min)", "Signal"]
        df = df[[col for col in df.columns if col in required_columns]]
        return df

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(load_and_clean_data, files))

    return results

def numerical_sort_key(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else 0

def folder_handling(folder_path):
    if folder_path is None:
        folder_path = input("Input folder location of converted .csv UHLPC files: ")

    # Clean the folder path by removing quotes
    folder_path = folder_path.replace('"', "").replace("'", "")

    # Retrieve and sort CSV files
    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".csv")],
        key=numerical_sort_key  # Ensure numerical_sort_key is defined or imported
    )
    return folder_path, csv_files

def windows_handling(windows):
    if not(windows):
        user_input = input(f"Enter the window bounds as two numbers separated by a comma (e.g., 10.5,20.0): ")
        try:
            lower, upper = map(float, user_input.split(","))
            windows = [lower, upper]
        except ValueError:
            print("Invalid input. Please enter two numbers separated by a comma.")
            windows = [10.5,20.0]  # Use default if invalid
    else:
        windows = [10.5,20.0]
    return windows

def hplc_integration_gen(folder_path=None, windows=True, peak_neighborhood_n=5, smoothing_window=12, smoothing_factor=3, gaus_iterations=4000, maximum_peak_amplitude=None, peak_boundary_derivative_sensitivity=0.01, peak_prominence=0.001):
    folder_path, csv_files = folder_handling(folder_path)
    print("Reading data...")
    data = read_data_concurrently(folder_path, csv_files)

    windows = windows_handling(windows)

    iref = True
    ref = None

    for df in data:
        sample_name = df["Sample Name"].iloc[0]
        analyzer = chromatopy_general.SignalAnalyzer(df, windows, gaus_iterations, sample_name, peak_neighborhood_n, smoothing_window,
                                  smoothing_factor, peak_boundary_derivative_sensitivity, peak_prominence,
                                  maximum_peak_amplitude, iref, ref)
        peaks, fig, ref_new, r_pressed, e_pressed = analyzer.run()
        if iref:
            ref = ref_new
            iref = False
        elif r_pressed:
            ref = peaks
            print(f"Reference peaks updated using {sample_name}.")
        elif e_pressed:
            print("You have exited your session.")
            break





