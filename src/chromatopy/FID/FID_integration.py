# ─── Standard Library ───────────────────────────────────────────────────────────
import os
import re
import sys
import shutil

# ─── Third-Party Libraries ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.widgets import TextBox, Button
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import json

# ─── PyQt5 GUI Toolkit ─────────────────────────────────────────────────────────
from PyQt5.QtCore    import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QLineEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QMessageBox)

# ─── Peak Integration ─────────────────────────────────────────────────────────
from .FID_Integration_functions import run_peak_integrator
from .FID_General import get_cluster_labels
from .manual_peak_integration import run_peak_integrator_manual
from .import_data import import_data

def integration(
        # Peak selection
        peak_labels=False, selection_method="nearest",
        # Data categorization
        categorized=None, 
        # Peak deconvolution 
        gaussian_fit_mode='single',  manual_peak_integration=False,
        # Peak integration parameters
        peak_neighborhood_n=3, smoothing_window=5, 
        smoothing_factor=3, gaus_iterations=1000, maximum_peak_amplitude=None, 
        peak_boundary_derivative_sensitivity=0.01, peak_prominence=0.01):
    """
    Main integration function for processing chromatographic samples.

    Parameters:
    - categorized: Dictionary of pre-categorized data. If None, raw data is imported.
    - selection_method: 'click' or 'nearest' for peak selection.
    - gaussian_fit_mode: 'single', 'multi', or 'both' for Gaussian fitting strategy.
    - peak_neighborhood_n: Maximum number of peaks in a neighborhood.
    - smoothing_window: Savitzky-Golay filter window size.
    - smoothing_factor: Savitzky-Golay polynomial order.
    - gaus_iterations: Max iterations for curve fitting.
    - maximum_peak_amplitude: Optional peak amplitude cap.
    - peak_boundary_derivative_sensitivity: Derivative threshold for boundary detection.
    - peak_prominence: Prominence threshold for peak finding.
    - peak_labels: If True, load peak label config from 'peak_labels.json'
    """
    
    # Handle predefined peak labels
    if peak_labels and manual_peak_integration:
        json_path = os.path.join(os.path.dirname(__file__), "peak_labels.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                peak_labels_data = json.load(f)
        else:
            raise FileNotFoundError(f"Expected peak_labels.json in {json_path}")
    else: peak_labels_data=None
    
    # Handle pre-categorized Data
    if categorized is not None:
        tqdm.write("Pre-categorized data.")
        result = import_data()
        unprocessed = result["unprocessed_samples"]
        if not unprocessed:
            return
        
        data = result['data_dict']
        time_column = result["time_column"]
        signal_column = result["signal_column"]
        folder_path = result["folder_path"]
        output_path = result["output_path"]
        figures_path = result["figures_path"]
        cluster_labels = get_cluster_labels(categorized)
        for i in cluster_labels:
            cluster_subset = {
                'Samples': {
                    key: value for key, value in data['Samples'].items()
                    if value.get('cluster') == i},
                'Integration Metadata': {}}
            FID_integration_backend(
                cluster_subset, time_column, signal_column,
                folder_path, output_path, figures_path,
                selection_method, gaussian_fit_mode,
                peak_neighborhood_n, smoothing_window,
                smoothing_factor, gaus_iterations,
                maximum_peak_amplitude, peak_boundary_derivative_sensitivity,
                peak_prominence,
                manual = manual_peak_integration,
                peak_labels=peak_labels_data)
    else:

        result = import_data()
        unprocessed = result["unprocessed_samples"]
        if not unprocessed:
            return
        data = result['data_dict']
        subset = {"Samples": {k: data["Samples"][k] for k in unprocessed},
          "Integration Metadata": {}}
        time_column = result["time_column"]
        signal_column = result["signal_column"]
        folder_path = result["folder_path"]
        output_path = result["output_path"]
        figures_path = result["figures_path"]
        FID_integration_backend(
            subset, time_column, signal_column,
            folder_path, output_path, figures_path,
            selection_method, gaussian_fit_mode,
            peak_neighborhood_n, smoothing_window, 
            smoothing_factor, gaus_iterations,
            maximum_peak_amplitude, peak_boundary_derivative_sensitivity,
            peak_prominence, manual = manual_peak_integration,
            peak_labels=peak_labels_data)
    
def print_no_samples_to_process():
        tqdm.write("All samples in this directory have been processed.")
        tqdm.write("To re-process sample entries, delete the samples with:")
        tqdm.write("    chromatopy.FID.delete_samples()")
        tqdm.write("Then rerun:")
        tqdm.write("    chromatopy.FID.integration()")
    
def FID_integration_backend(data, time_column, signal_column, folder_path, 
                            output_path, figures_path, sm, gaussian_fit_mode,
                            peak_neighborhood_n=3, smoothing_window=35, smoothing_factor=3, 
                            gaus_iterations=4000, maximum_peak_amplitude=None, 
                            peak_boundary_derivative_sensitivity=0.01, peak_prominence=1, 
                            manual=False, peak_labels=None):
    
    # Get unprocessed samples only
    unprocessed_keys = [k for k in data["Samples"].keys() if 'Processed Data' not in data["Samples"][k].keys()]

    # Identify peak locations
    if manual and peak_labels is not None:
        tqdm.write("Using stored peak labels for manual integration.")
        data['Integration Metadata'] = {
            "peak dictionary": peak_labels["Peak Labels"],
            "x limits": peak_labels["x limits"],
            "time_column": time_column,
            "signal_column": signal_column}
    else:
        tqdm.write("Click the location of peaks and enter the chain length of interest (e.g., C22).\nUse 'shift+delete' to remove the last peak.\n'Select 'Finished' once satisfied.")
        app = QApplication.instance() or QApplication(sys.argv)
        first_key = unprocessed_keys[0]
        time = data['Samples'][first_key]['Raw Data'][time_column]
        signal = data['Samples'][first_key]['Raw Data'][signal_column]

        if sm == "nearest":
            peak_positions, _ = find_peaks(signal)
        elif sm == "click":
            peak_positions = None

        peak_identifier = FID_Peak_ID(x=time, y=signal, selection_method=sm, peak_positions=peak_positions)
        app.exec_()

        data['Integration Metadata'] = {
            "peak dictionary": peak_identifier.result,
            "time_column": time_column,
            "signal_column": signal_column}

    for key in tqdm(unprocessed_keys, desc="Integrating samples", unit="sample", mininterval=0, maxinterval=0):
        if "Integratoin Result" in data['Samples'][key].keys():
            tqdm.write(f"{key} already processed")
            continue

        if manual:
            run_peak_integrator_manual(data, key, gi=gaus_iterations,
                                       pk_sns=peak_boundary_derivative_sensitivity,
                                       smoothing_params=[smoothing_window, smoothing_factor],
                                       max_peaks_for_neighborhood=peak_neighborhood_n,
                                       fp=figures_path,
                                       gaussian_fit_mode=gaussian_fit_mode)
        else:
            run_peak_integrator(data, key, gi=gaus_iterations,
                                pk_sns=peak_boundary_derivative_sensitivity,
                                smoothing_params=[smoothing_window, smoothing_factor],
                                max_peaks_for_neighborhood=peak_neighborhood_n,
                                fp=figures_path,
                                gaussian_fit_mode=gaussian_fit_mode)
        data = round_dict_floats(data)
        existing_data = load_json(output_path)
        if existing_data:
            for sample_name, sample_data in data["Samples"].items():
                existing_data["Samples"][sample_name] = sample_data
        else:
            existing_data = data

        save_json({'data_dict':existing_data}, output_path)
        output_csv(existing_data, output_path)
    # return data

def output_csv(data, output_directory):
    """
    Generate two CSV files:
    1. output_peak_areas.csv: contains median peak areas per sample.
    2. output_retention_times.csv: contains retention times per sample.

    Parameters
    ----------
    data : dict
        Data dictionary containing processed chromatographic data.
    output_directory : str
        Directory to save the output CSVs.

    Returns
    -------
    df_areas : pandas.DataFrame
        DataFrame of peak area medians per sample.
    df_ret_times : pandas.DataFrame
        DataFrame of retention times per sample.
    """
    # Collect all unique peak labels across samples
    all_peaks = set()
    for sample in data['Samples'].values():
        processed = sample.get("Processed Data", {})
        all_peaks.update(processed.keys())

    all_peaks = sorted(all_peaks)

    peak_area_rows = []
    retention_time_rows = []

    for sample_name, sample in data['Samples'].items():
        row_area = {"Lab ID": sample_name}
        row_ret = {"Lab ID": sample_name}
        processed = sample.get("Processed Data", {})
        for peak in all_peaks:
            peak_data = processed.get(peak, None)
            if peak_data and isinstance(peak_data, dict):
                row_area[peak] = peak_data.get("Peak Area - median", np.nan)
                row_ret[peak] = peak_data.get("Retention Time", np.nan)
            else:
                row_area[peak] = np.nan
                row_ret[peak] = np.nan
        peak_area_rows.append(row_area)
        retention_time_rows.append(row_ret)

    # Convert to DataFrames
    df_areas = pd.DataFrame(peak_area_rows)
    df_ret_times = pd.DataFrame(retention_time_rows)

    # Save as CSVs
    os.makedirs(output_directory, exist_ok=True)
    area_path = os.path.join(output_directory, "output_peak_areas.csv")
    rt_path = os.path.join(output_directory, "output_retention_times.csv")
    df_areas.to_csv(area_path, index=False)
    df_ret_times.to_csv(rt_path, index=False)

def convert_dataframes_to_dicts(obj):
    """
    Walks any nested structure of dicts/lists and converts
    any pandas.DataFrame it finds into a plain dict of lists.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, dict):
        return {k: convert_dataframes_to_dicts(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dataframes_to_dicts(v) for v in obj]
    else:
        return obj

def save_json(container, output_path):
    # “container” is the dict you get back from import_data()
    data_dict = container["data_dict"]

    # convert _all_ DataFrames under data_dict → dicts of lists
    cleanable = convert_dataframes_to_dicts(data_dict)

    js_file = os.path.join(output_path, "FID_output.json")
    os.makedirs(os.path.dirname(js_file), exist_ok=True)
    with open(js_file, "w") as f:
        json.dump(clean_for_json(cleanable), f, indent=4)

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


# def create_output_folders(folder_path):
#     """
#     Creates a 'chromatoPy output' folder inside the given folder_path.
#     If it already exists, deletes it and recreates it.
#     Also creates a nested 'Figures' subfolder.

#     Returns
#     -------
#     output_path : str
#         Path to 'chromatoPy output' folder.
#     figures_path : str
#         Path to 'chromatoPy output/Figures' folder.
#     """
#     output_path = os.path.join(folder_path, "chromatoPy output")
#     figures_path = os.path.join(output_path, "Figures")
    
#     os.makedirs(output_path, exist_ok=True)
#     os.makedirs(figures_path, exist_ok=True)

#     return output_path, figures_path

def clean_for_json(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.ndarray, pd.Series, list, tuple)):
        return [clean_for_json(el) for el in obj]
    elif isinstance(obj, pd.DataFrame): 
        return obj.to_dict(orient="list")
    elif isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)
        
# def parse_metadata_block(raw_text):
#     """
#     Parse chromatogram metadata string into a nested dictionary.
#     """
#     lines = raw_text.strip().split("\n")
#     result = {}
#     current_section = None

#     for line in lines:
#         if not line.strip():
#             continue  # skip empty lines

#         parts = line.split("\t")
#         parts = [p.strip() for p in parts if p.strip()]

#         if len(parts) == 1:
#             # This is likely a section header like "Injection Information:"
#             section = parts[0].rstrip(":")
#             result[section] = {}
#             current_section = section
#         elif len(parts) == 2:
#             key, value = parts
#             if current_section:
#                 result[current_section][key] = value
#             else:
#                 result[key] = value
#         else:
#             # Unhandled line structure
#             tqdm.write("Skipping malformed line:", line)

#     return result
 
def merge_existing_jsons(data, output_path):
    """
    Function Identifies previously processed data (.json files) in working
    directory, selects samples from this previous dataset that are not in
    the raw .txt files, and adds these samples to the new dictionary.
    --- Requires heavy modificatoin to properly retain model information.

    Parameters
    ----------
    data : dict
        Imported data from .txt file. Contains raw data
    output_path : str
        Location of output directory.

    Returns
    -------
    data : dict
        Dataset.
    """
    # output_path, figures_path = create_output_folders(folder_path)
    existing_data = load_json(output_path)
    new_samples = {}
    if existing_data is not None:
        for sample_name, sample in data["Samples"].items():
            if sample_name not in existing_data["Samples"]:
                existing_data["Samples"][sample_name] = sample
        data = existing_data
    save_json(data, output_path)
    return data


# def check_existing_jsons(data_dict, output_path):
#     """
#     Merge any new samples in data_dict into an existing JSON on disk.
#     Returns: (merged_dict, unprocessed_keys)
#       - merged_dict: either the existing JSON plus new samples, or your original data_dict
#       - unprocessed_keys: list of sample-names that came in fresh
#     """
#     existing = load_json(output_path)
#     # no file on disk → nothing processed yet
#     if existing is None:
#         unprocessed = list(data_dict["Samples"].keys())
#         return data_dict, unprocessed

#     # else: there was an existing JSON
#     before = set(existing["Samples"].keys())
#     # add any brand-new samples
#     for name, sample in data_dict["Samples"].items():
#         if name not in existing["Samples"]:
#             existing["Samples"][name] = sample

#     # figure out which ones we just added
#     after = set(existing["Samples"].keys())
#     unprocessed = list(after - before)

#     return existing, unprocessed


class FID_Peak_ID:
    def __init__(self, x, y, selection_method, peak_positions=None):
        self.x = pd.Series(x)
        self.y = pd.Series(y)
        self.selection_method = selection_method
        self.lines = []
        self.labels = []
        self.positions = set()
        self.peak_dict = {}
        self.peak_order = []

        if peak_positions is None:
            self.peak_positions = []
        else:
            self.peak_positions = list(peak_positions)

        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.plot(self.x, self.y)

        # TextBoxes for axis limits
        self.textbox_minx_ax = self.fig.add_axes([0.15, 0.02, 0.1, 0.04])
        self.textbox_maxx_ax = self.fig.add_axes([0.3, 0.02, 0.1, 0.04])
        self.textbox_miny_ax = self.fig.add_axes([0.45, 0.02, 0.1, 0.04])
        self.textbox_maxy_ax = self.fig.add_axes([0.6, 0.02, 0.1, 0.04])
        self.textbox_minx = TextBox(self.textbox_minx_ax, 'X0')
        self.textbox_maxx = TextBox(self.textbox_maxx_ax, 'X1')
        self.textbox_miny = TextBox(self.textbox_miny_ax, 'Y0')
        self.textbox_maxy = TextBox(self.textbox_maxy_ax, 'Y1')

        # Initialize limits
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        self.textbox_minx.set_val(str(round(xmin, 1)))
        self.textbox_maxx.set_val(str(round(xmax, 1)))
        self.textbox_miny.set_val(str(round(ymin, 1)))
        self.textbox_maxy.set_val(str(round(ymax, 1)))

        # Connect axis limit updates
        for box in [self.textbox_minx, self.textbox_maxx, self.textbox_miny, self.textbox_maxy]:
            box.on_submit(self.update_limits)

        # Finish button
        self.button_ax = self.fig.add_axes([0.85, 0.02, 0.1, 0.04])
        self.finish_button = Button(self.button_ax, 'Finished')
        self.finish_button.on_clicked(self.finish)

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()

        # Focus - permits seeing key events
        self.fig.canvas.setFocusPolicy(Qt.StrongFocus)
        self.fig.canvas.setFocus()

    def on_click(self, event):
            if event.inaxes != self.ax:
                return
            # x_click = round(event.xdata, 5)
            # if x_click in self.positions:
            #     return

            if self.selection_method == "nearest" and self.peak_positions:
                raw_x = event.xdata
                # find the peak position closest to where they clicked
                # x_click = min(self.x[self.peak_positions], key=lambda xp: abs(xp - raw_x))
                peak_times = self.x.iloc[self.peak_positions].to_numpy()
                x_click = min(peak_times, key=lambda t: abs(t - raw_x))
            else:
                x_click = round(event.xdata, 5)

            if x_click in self.positions:
                return

            # 1) draw the line immediately (so user sees it)
            line = self.ax.axvline(x_click, color='red', linestyle='--', alpha=0.7)
            self.lines.append(line)

            # 2) ask for the label via our Qt dialog
            prompt = f"Label for peak at x = {x_click:.2f}"
            dlg = LabelDialog(prompt=prompt, initial="peak", parent=self.fig.canvas)
            if dlg.exec_() == QDialog.Accepted:
                text = dlg.value()
                # duplicate‐name check
                if text in self.peak_dict:
                    QMessageBox.warning(
                        self.fig.canvas, "Duplicate label",
                        f"'{text}' already exists—please pick another name."
                    )
                    # undo that line
                    self.lines.pop().remove()
                    return

                # 3) record & draw the annotation
                self.positions.add(x_click)
                self.peak_order.append(text)
                self.peak_dict[text] = x_click

                y_top = self.ax.get_ylim()[1]
                txt = self.ax.text(
                    x_click + 0.05, y_top * 0.95, text, #rotation=90,
                    verticalalignment='top', horizontalalignment='left',
                    color='red', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.5)
                )
                self.labels.append(txt)
                self.fig.canvas.draw_idle()

            else:
                # user cancelled → remove that line
                self.lines.pop().remove()
                self.fig.canvas.draw_idle()

    def on_key(self, event):
        qt_ev = getattr(event, "guiEvent", None)
        if not qt_ev:
            return

        keycode = qt_ev.key()
        # Qt.Key_Backspace = 16777219, Qt.Key_Delete = 16777223
        if (qt_ev.modifiers() & Qt.ShiftModifier) and keycode in (Qt.Key_Backspace, Qt.Key_Delete):
            if not self.peak_order:
                return

            # 1) remove last vertical line
            line = self.lines.pop()
            line.remove()

            # 2) remove last text annotation
            txt = self.labels.pop()
            txt.remove()

            # 3) clean up bookkeeping
            last_label = self.peak_order.pop()
            x_removed = self.peak_dict.pop(last_label)
            self.positions.discard(x_removed)

            # 4) redraw
            self.fig.canvas.draw_idle()

    def update_limits(self, _):
        # Update axis limits from TextBox values
        try:
            xmin = float(self.textbox_minx.text)
            xmax = float(self.textbox_maxx.text)
            ymin = float(self.textbox_miny.text)
            ymax = float(self.textbox_maxy.text)
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.fig.canvas.draw_idle()
        except ValueError:
            tqdm.write('Invalid axis limits entered.')
        finally:
            # force-release any mouse grab so next click is clean
            try:
                self.fig.canvas.release_mouse(self.ax)
            except Exception:
                pass

    def finish(self, event):
        # Store the peaks dict so callers can grab it
        self.result = dict(self.peak_dict)
        # Close the plot
        plt.close(self.fig)
        # Quit the Qt event loop so exec_() returns
        QApplication.instance().quit()

class LabelDialog(QDialog):
    def __init__(self, prompt="Enter peak label:", initial="peak", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Peak Label")
        # Prompt text
        lbl = QLabel(prompt, self)
        # The line‐edit
        self.edit = QLineEdit(self)
        self.edit.setText(initial)
        self.edit.selectAll()           # select all so typing replaces
        # OK button
        ok = QPushButton("OK", self)
        ok.clicked.connect(self.accept)
        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(lbl)
        layout.addWidget(self.edit)
        layout.addWidget(ok)
        self.setLayout(layout)

    def value(self):
        return self.edit.text().strip()

def round_dict_floats(data, default_decimals=4):
    """
    Recursively round all floats in a nested dictionary or list structure to a fixed number of decimals.
    Handles DataFrames by rounding numeric columns.
    Does not round stringified DataFrame representations.

    Parameters:
    - data: dict or list
    - default_decimals: int, number of decimal places to round to
    """
    def process_df(df):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(default_decimals)
        return df

    def recursive_round(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if k == "Raw Data" and isinstance(v, pd.DataFrame):
                    new_dict[k] = process_df(v.copy())
                else:
                    new_dict[k] = recursive_round(v)
            return new_dict
        elif isinstance(obj, list):
            return [recursive_round(v) for v in obj]
        elif isinstance(obj, pd.DataFrame):
            return process_df(obj.copy())
        elif isinstance(obj, (float, np.floating)):
            return round(obj, default_decimals)
        else:
            return obj

    return recursive_round(data)