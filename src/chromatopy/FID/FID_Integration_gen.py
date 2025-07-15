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
from matplotlib.widgets import Button, TextBox
from scipy.signal import find_peaks
from tqdm import tqdm
import json

# ─── PyQt5 GUI Toolkit ─────────────────────────────────────────────────────────
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QLineEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QMessageBox)

# ─── Peak Integration ─────────────────────────────────────────────────────────
from .FID_Integration_functions import *

def FID_integration_WIP(df, x_col, y_col, timing = None, peak_neighborhood_n=5, smoothing_window=11, smoothing_factor=3, gaus_iterations=4000, peak_boundary_derivative_sensitivity=0.01, peak_prominence=0.001, selection_method="nearest"):
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce').fillna(0)

    x = df[x_col]
    y = df[y_col]

    if timing is None:
        app = QApplication.instance() or QApplication(sys.argv)
        if selection_method == "nearest":
            peak_positions, _ = find_peaks(y)
        else:
            peak_positions = None

        peak_identifier = FID_Peak_ID(x=x, y=y, selection_method=selection_method, peak_positions=peak_positions)
        app.exec_()

        data = {"Samples": {"InMemory": {"raw data": df}},
                "Integration Metadata": {"peak dictionary": peak_identifier.result, "time_column": x_col, "signal_column": y_col}}
    else:
        timing = validate_timing(timing)
        timing_df = pd.read_csv(timing)
        col_lbl = input("Enter your header for labels column: ")
        col_time = input("Enter your header for timings column: ")
        peak_dict = timing_df.set_index(col_lbl).to_dict()[col_time]
        data = {"Samples": {"InMemory": {"raw data": df}},
                "Integration Metadata": {"peak dictionary": peak_dict, "time_column": x_col, "signal_column": y_col}}

    run_peak_integrator(
        data,
        key="InMemory",
        gi=gaus_iterations,
        pk_sns=peak_boundary_derivative_sensitivity,
        smoothing_params=[smoothing_window, smoothing_factor],
        max_peaks_for_neighborhood=peak_neighborhood_n,
        fp=None
    )

    return data["Samples"]["InMemory"]["Integration"]

 # Assume 'timing' is a path to the csv containing peak timing
        # 1. Read csv as dataframe. Include error check for pathway (e.g., does path exist? Is dataset correct type?
                # If an error exists, reprompt the user for pathway
        # 2. Create dictionary from labels and peak timing
        # 3. Add dictionary to sample data

def validate_timing(timing):
    while True:
        try:
            if type(timing) != str:
                raise TypeError()
            if not (timing.endswith('.csv')):
                raise ValueError()
            with open(timing, 'r'):
                pass
        except FileNotFoundError:
            print("The timing file does not exist.")
        except ValueError:
            print("The timing file is not a .csv file")
        except TypeError:
            print("The timing file path is not a string")
        except Exception as e:
            print(f"Unable to use file: {e}")
        else:
            return timing
        timing = input("Enter a valid .csv file path: ")

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
                x_click + 0.05, y_top * 0.95, text,  # rotation=90,
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
        self.edit.selectAll()  # select all so typing replaces
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
