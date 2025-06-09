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
from FID_Integration_functions import *

def FID_integration(folder_path=None, peak_neighborhood_n=5, smoothing_window=11, smoothing_factor=3, gaus_iterations=4000, maximum_peak_amplitude=None, peak_boundary_derivative_sensitivity=0.01, peak_prominence=0.001, selection_method="nearest"):
    # Import Data
    data, no_time_col, no_signal_col, time_column, signal_column = import_data(folder_path)
    output_path, figures_path = create_output_folders(folder_path)
    
    # Instructions
    print("Click the location of peaks and enter the chain length of interest (e.g., C22).\nUse 'shift+delete' to remove the last peak.\n'Select 'Finished' once satisfied.")

    # Identify peak locations
    app = QApplication.instance() or QApplication(sys.argv)
    time = data['Samples'][list(data['Samples'].keys())[0]]['raw data'][time_column]
    signal = data['Samples'][list(data['Samples'].keys())[0]]['raw data'][signal_column]
    
    # Identify peak positions
    if selection_method == "nearest":
        peak_positions, _ = find_peaks(signal)
    elif selection_method == "click":
        peak_positions = None
    peak_identifier = FID_Peak_ID(x=time, y = signal, selection_method=selection_method, peak_positions=peak_positions)
    app.exec_()
    data['Integration Metadata'] = {}
    data['Integration Metadata']['peak dictionary'] = peak_identifier.result
    data['Integration Metadata']['time_column'] = time_column
    data['Integration Metadata']['signal_column'] = signal_column
    # print(f"Results from Integradtion Metadata: {peak_identifier.result}")
    
    # Integrate peaks
    for key in tqdm(data['Samples'].keys(), desc="Integrating samples", unit="sample"):
        # peak_timing = list(data['Integration Metadata'].values())
        tqdm.write(f"Processing: {key}")
        run_peak_integrator(data, key, gi = gaus_iterations, pk_sns = peak_boundary_derivative_sensitivity, 
                            smoothing_params=[smoothing_window, smoothing_factor], max_peaks_for_neighborhood = peak_neighborhood_n, fp=figures_path)
    js_file = f"{output_path}/FID_output.json"
    os.makedirs(os.path.dirname(js_file), exist_ok=True)
    with open(js_file, "w") as f:
        json.dump(clean_for_json(data), f, indent=4)


def create_output_folders(folder_path):
    """
    Creates a 'chromatoPy output' folder inside the given folder_path.
    If it already exists, deletes it and recreates it.
    Also creates a nested 'Figures' subfolder.

    Returns
    -------
    output_path : str
        Path to 'chromatoPy output' folder.
    figures_path : str
        Path to 'chromatoPy output/Figures' folder.
    """
    output_path = os.path.join(folder_path, "chromatoPy output")
    figures_path = os.path.join(output_path, "Figures")

    # Remove output folder if it already exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Create output and figures subfolders
    os.makedirs(figures_path)

    return output_path, figures_path

def clean_for_json(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.ndarray, pd.Series, list, tuple)):
        return [clean_for_json(el) for el in obj]
    elif isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    else:
        try:
            json.dumps(obj)  # test if serializable
            return obj
        except (TypeError, OverflowError):
            return str(obj)  # fallback

def import_data(folder_path=None):
    # Prompt user if no path is provided
    if folder_path is None:
        folder_path = input("Provide folder containing .txt files: ")
        folder_path = folder_path.strip('\'"')
    if not folder_path:
        print("No folder selected. Aborting.")
        raise SystemExit

    # List all .txt files
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".txt")]
    if not txt_files:
        print(f"No .txt files found in {folder_path}. Aborting.")
        raise SystemExit

    no_time_col = []
    no_signal_col = []
    data_dict = {}
    data_dict["Samples"] = {}

    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Find the line where the table begins
        table_start = None
        for i, line in enumerate(lines):
            if re.search(r'(?i)^\s*Chromatogram Data\s*:', line):
                table_start = i + 1  # The actual table header is the next line
                break

        if table_start is None or table_start + 1 >= len(lines):
            print(f"Could not find table start in {filename}")
            continue

        # Read headers
        headers = lines[table_start].strip().split('\t')
        data_lines = lines[table_start + 1:]

        # Read into DataFrame
        try:
            df = pd.DataFrame([l.strip().split('\t') for l in data_lines if l.strip() != ''], columns=headers)
        except Exception as e:
            print(f"Failed to parse table in {filename}: {e}")
            continue

        # Header matching
        time_keywords = ['time', 'min', 'sec', 'second', 'minute']
        signal_keywords = ['signal', 'value', 'intensity', 'amplitude', '(pa)', '(a)']
        headers = lines[table_start].strip().split('\t')
        header_map = {h.lower(): h for h in headers}
        
        time_column = next((header_map[h] for h in header_map if any(key in h for key in time_keywords)), None)
        has_time = time_column is not None
        signal_column = next((header_map[h] for h in header_map if any(key in h for key in signal_keywords)), None)
        has_signal = signal_column is not None
        
        # Numeric dataframe
        df[time_column] = pd.to_numeric(df[time_column], errors='coerce')
        df[signal_column] = pd.to_numeric(df[signal_column], errors='coerce')
        df[signal_column] = df[signal_column].fillna(0)
        
        if not has_time:
            no_time_col.append(filename)
        if not has_signal:
            no_signal_col.append(filename)

        # Metadata is everything before the table
        metadata = ''.join(lines[:table_start - 1])
        # Store in dictionary
        data_dict['Samples'][filename.replace(".txt", "")] = {
            "metadata": metadata,
            "raw data": df}

    print(f"Found {len(txt_files)} .txt files.")
    if no_time_col:
        print("Files missing time column:", no_time_col)
    if no_signal_col:
        print("Files missing signal column:", no_signal_col)

    return data_dict, no_time_col, no_signal_col, time_column, signal_column


class FID_Peak_ID:
    def __init__(self, x, y, selection_method, peak_positions=None):
        self.x = x
        self.y = y
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
            print('Invalid axis limits entered.')

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
# %%
dat = FID_integration(folder_path='/Users/gerard/Desktop/Emanda_FID/test' , selection_method="nearest")
