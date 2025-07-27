import os
from . import chromatopy_gen
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import re
import json
import logging

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

def output_handling(folder_path):
    output_folder = os.path.join(folder_path, "Output_chromatoPy")
    os.makedirs(output_folder,exist_ok=True)
    figures_folder = os.path.join(output_folder, "Figures_chromatoPy")
    os.makedirs(figures_folder, exist_ok=True)
    json_folder = os.path.join(output_folder, "Individual Samples")
    os.makedirs(json_folder, exist_ok=True)
    save_path = os.path.join(output_folder, "Output.csv")
    paths = [figures_folder,json_folder,save_path]
    return paths

def abortion_handling(output, paths, abort_sample):
    output_samples = output["Sample Name"].tolist()
    output_samples.sort(key = numerical_sort_key)

    if abort_sample in output_samples and abort_sample == output_samples[-1]:
        samples_wanted = output_samples
        samples_removed = []
    else:
        idx = output_samples.index(abort_sample)
        samples_wanted = output_samples[:idx+1]
        samples_removed = output_samples[idx+1:]

    figs = []
    for fig_file in os.listdir(paths[0]):
        figs.append(fig_file.removesuffix("_fig.png"))
    jsons = []
    for json_file in os.listdir(paths[1]):
        jsons.append(json_file.removesuffix(".json"))
    for figf in figs:
        if figf not in samples_wanted:
            os.remove(os.path.join(paths[0], figf + "_fig.png"))
    for jsonf in jsons:
        if jsonf not in samples_wanted:
            os.remove(os.path.join(paths[1], jsonf + ".json"))

    for sample in samples_removed:
        output.drop(output[output["Sample Name"] == sample].index, inplace=True)

def hplc_integration_gen(folder_path=None, compounds=None, window_bounds=[10.5,20], headers=["RT (min)","Signal"], peak_neighborhood_n=5, smoothing_window=12, smoothing_factor=3, gaus_iterations=4000, maximum_peak_amplitude=None, peak_boundary_derivative_sensitivity=0.01, peak_prominence=0.001):
    folder_path, csv_files = folder_handling(folder_path)
    print("Reading data...")
    logging.info("Data is being read from the inputted folder.")
    data = read_data_concurrently(folder_path, csv_files, headers)

    iref = True
    aborted = compound_error = False
    ref = None

    paths = output_handling(folder_path)

    if os.path.exists(paths[2]):
        output = pd.read_csv(paths[2])
    else:
        output = pd.DataFrame(columns=['Sample Name'] + compounds)

    logging.info("Output directory has been setup.")

    for df in data:
        sample_name = df["Sample Name"].iloc[0]

        if sample_name in output["Sample Name"].values:
            continue

        logging.info(f"{sample_name} is being processed.")

        analyzer = chromatopy_gen.SignalAnalyzer(df, compounds, window_bounds, headers, gaus_iterations, sample_name, peak_neighborhood_n, smoothing_window,
                                  smoothing_factor, peak_boundary_derivative_sensitivity, peak_prominence,
                                  maximum_peak_amplitude, iref, ref)
        peaks, fig, ref_new, r_pressed, e_pressed = analyzer.run()

        if 'areas' not in peaks:
           areas = []
        else:
            areas = peaks['areas']
            rts = peaks['rts']

        if compounds is None or len(compounds) != len(areas):
            compound_error = True
            break

        odf = pd.DataFrame(list(zip(areas, rts)), columns=['areas', 'rts'])
        odf.sort_values(by="rts", inplace=True)
        odf['compounds'] = compounds
        output.loc[len(output)] = [sample_name] + odf["areas"].tolist()

        # Figures saving process
        fig_path = os.path.join(paths[0], f"{sample_name}_fig.png")
        fig.savefig(fig_path)

        # Peak structure saving process
        json_path = os.path.join(paths[1], f"{sample_name}.json")
        with open(json_path, "w", encoding="utf-8") as outfile:
            json.dump(peaks, outfile, indent=3)

        if iref:
            ref = ref_new
            iref = False
        elif r_pressed:
            ref = peaks
            msg = f"Reference peaks updated using {sample_name}."
            logging.info(msg)
            print(msg)
        elif e_pressed:
            abortion_handling(output, paths, sample_name)
            aborted = True
            break

    output.sort_values(
        by="Sample Name",
        key=lambda col: col.map(numerical_sort_key),
        inplace=True
    )

    output.to_csv(paths[2], index=False)

    if compound_error:
        print(f"The number of peak clicks weren't equal to the number of compounds for sample: {sample_name}")
        return ("compound_error", sample_name)
    elif aborted:
        print(f"Integration aborted by user at sample: {sample_name}")
        return ("aborted", sample_name)
    else:
        print("HPLC integration completed successfully.")
        return ("success", sample_name)






