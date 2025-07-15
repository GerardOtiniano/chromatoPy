import os
from . import chromatopy_gen
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import re

def read_data_concurrently(folder_path, files, headers):
    def load_and_clean_data(file):
        full_path = os.path.join(folder_path, file)
        df = pd.read_csv(full_path)

        df["Sample Name"] = os.path.basename(file)[:-4]

        # if "RT(minutes) - NOT USED BY IMPORT" in df.columns:
        #     df.rename(columns={"RT(minutes) - NOT USED BY IMPORT": "RT (min)"}, inplace=True)

        df.columns = [col[:-2] if col.endswith(".0") else col for col in df.columns]

        required_columns = ["Sample Name", headers[0], headers[1]]
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

def hplc_integration_gen(folder_path=None, compounds=None, window_bounds=[10.5,20], headers=["RT (min)","Signal"], peak_neighborhood_n=5, smoothing_window=12, smoothing_factor=3, gaus_iterations=4000, maximum_peak_amplitude=None, peak_boundary_derivative_sensitivity=0.01, peak_prominence=0.001):
    folder_path, csv_files = folder_handling(folder_path)
    print("Reading data...")
    data = read_data_concurrently(folder_path, csv_files, headers)

    iref = True
    ref = None
    output = pd.DataFrame(columns="column names")
    for df in data:
        sample_name = df["Sample Name"].iloc[0]
        analyzer = chromatopy_gen.SignalAnalyzer(df, compounds, window_bounds, headers, gaus_iterations, sample_name, peak_neighborhood_n, smoothing_window,
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
            print(f"Integration aborted by user at sample: {sample_name}")
            return "aborted"

'''
Dataframe
first column: sample names
first row: peak names
A1 empty (first cell)
each column is a peak
each row is a sample
and cell contains peak area for that sample for that peak

1) Take peak areas from peak results, put into dataframe first columns
2) Take rts from peak results, put into second column
3) sort by second column (rts)
4) add names/compounds to third column 
5) this dataframe goes into a new dataframe where first row: gets all third column values, row: sample name, areas.

1) Save peaks structure as json file.

1) Figures folder storing every figure
'''





