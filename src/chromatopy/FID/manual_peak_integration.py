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
from FID_Integration_functions import run_peak_integrator, smoother, baseline, find_valleys, find_peak_neighborhood_boundaries, fit_gaussians
    
def run_peak_integrator_manual(data, key, gi, pk_sns, smoothing_params, max_peaks_for_neighborhood, fp, gaussian_fit_mode):
    # Setup data
    md = data['Integration Metadata']
    x = data['Samples'][key]['Raw Data'][md['time_column']]
    y = data['Samples'][key]['Raw Data'][md['signal_column']]
    # Subset to x limits: either deduce from a dict of times, or fall back to explicit x-limits
    pdict = md['peak dictionary']
    if isinstance(pdict, dict):
         peak_times = list(pdict.values())
         labels     = list(pdict.keys())
         xmin, xmax = min(peak_times) - 0.4, max(peak_times) + 0.4
    else:
        # we're given just a list of labels
        labels     = pdict
        x0, x1     = md['x limits']
        xmin, xmax = x0 - 0.4, x1 + 0.4
    mask = (x >= xmin) & (x <= xmax)
    xdata = x[mask].reset_index(drop=True)
    ydata = y[mask].reset_index(drop=True)
    ydata = smoother(ydata, *smoothing_params)
    ydata = pd.Series(ydata, index=xdata.index)
    ydata[ydata < 0] = 0
    base, min_peak_amp = baseline(xdata, ydata, deg=500, max_it=1000, tol=1e-4)
    y_bcorr = ydata - base
    peak_indices, peak_properties = find_peaks(y_bcorr, height=min_peak_amp)#min_peak_amp)
    valleys = find_valleys(y_bcorr, peak_indices)
    peak_labels = labels

    # return data
    peak_selector = ManualPeakIntegrator(
        xdata,                   # as a numpy array
        y_bcorr,
        peak_indices,                       # pass the raw peak indices
        peak_properties,                    # needed for neighborhood fitting
        valleys,
        peak_labels,
        smoothing_params,
        pk_sns,
        gi,
        gaussian_fit_mode)
    app = QApplication.instance() or QApplication(sys.argv)
    app.exec_()
    
    
    if peak_selector.force_exit:
        tqdm.write("Manual integration was forcefully exited by the user.")
        raise SystemExit  # or return None, or raise a custom exception
    
    # Save output
    data['Samples'][key]['Processed Data'] = peak_selector.processed_data
    peak_selector.fig.savefig(str(fp) + f"/{key}.png", dpi=300)
    plt.close(peak_selector.fig)
    return data


class ManualPeakIntegrator:
    def __init__(self,
                 x, y,
                 peaks,           # <— list/array of peak indices
                 peak_properties,
                 valleys,
                 labels,
                 smoothing_params,
                 pk_sns,
                 gi,
                 gaussian_fit_mode):
        self.x, self.y = x, y
        self.valleys = valleys
        self.peaks = np.asarray(peaks)
        self.peak_properties = peak_properties
        self.labels = labels
        self.smoothing_params = smoothing_params
        self.pk_sns = pk_sns
        self.gi = gi
        self.gaussian_fit_mode = gaussian_fit_mode

        self.index = 0
        self.processed_data = {}
        self.artists_stack = []
        self.click_tolerance = 10/60  # 10 seconds in minutes
        self.finished = False

        # figure + data plot
        self.fig, self.ax = plt.subplots()
        self.ax.axhline(0, c='k')
        self.ax.plot(self.x, self.y, c='k', alpha = 0.6)
        self.text = self.ax.text(
            0.5, 0.95, f"Click peak for: {self.labels[self.index]}",
            transform=self.ax.transAxes, ha='center')
        self.ax.set_xlabel('Retention Time (min)')
        self.ax.set_ylabel("Value (pA)")
        # “Finished” button
        btn_ax = self.fig.add_axes([0.82, 0.02, 0.15, 0.05])
        self.finish_button = Button(btn_ax, "Finished")
        # Now finish accepts the event
        self.finish_button.on_clicked(self.finish)
        
        # Exit button
        exit_ax = self.fig.add_axes([0.82, 0.94, 0.15, 0.05])
        self.exit_button = Button(exit_ax, "Exit")
        self.exit_button.on_clicked(self.exit_program)
        
        # Flag for termination
        self.force_exit = False

        # connect events
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.cid_key   = self.fig.canvas.mpl_connect("key_press_event",   self.on_key)

    def exit_program(self, event=None):
        """Triggered when the Exit button is clicked."""
        self.force_exit = True
        self.text.set_text("Exiting...")
        self.fig.canvas.draw()
        QApplication.quit()
        plt.close(self.fig)
        
    def onclick(self, event):
         if event.inaxes != self.ax:
             return
    
         click_time = event.xdata
         peak_times = self.x.to_numpy()[self.peaks]
         dists = np.abs(peak_times - click_time)
         best = dists.argmin()
    
         # if the nearest real peak is > tolerance, treat as “no peak”
         if dists[best] <= self.click_tolerance:
             peak_idx = int(self.peaks[best])
         else:
             # no valid peak → grey dashed line & record NaN
             line = self.ax.axvline(click_time, color='grey', linestyle='--')
             self.artists_stack.append([line])
             self.processed_data[self.labels[self.index]] = {'Values': [np.nan]}
             self._advance_prompt()
             return
         drawn = []
         try:
             if self.gaussian_fit_mode in {"multi","both"}:
                 _, _, neigh = find_peak_neighborhood_boundaries(
                     self.x, self.y, self.peaks, self.valleys,
                     peak_idx, self.pk_sns,
                     peak_properties=self.peak_properties,
                     gi=self.gi,
                     smoothing_params=self.smoothing_params,
                     pk_sns=self.pk_sns)
             else:
                 neigh = [peak_idx]
             x_fit, y_fit, _, area_ensemble, model_params = fit_gaussians(
                 self.x, self.y, peak_idx, neigh,
                 self.smoothing_params, self.pk_sns,
                 gi=self.gi,
                 mode=self.gaussian_fit_mode)
             poly = self.ax.fill_between(x_fit, 0, y_fit, color='red', alpha=0.4)
             drawn.append(poly)
             self.processed_data[self.labels[self.index]] = {
                 'Area Ensemble': list(area_ensemble),
                 'Model Parameters': model_params}
    
         except Exception as e:
             tqdm.write(f"[Manual Warning] Failed to fit {self.labels[self.index]}: {e}")
             line = self.ax.axvline(click_time, color='grey', linestyle='--')
             drawn.append(line)
             self.processed_data[self.labels[self.index]] = {'Values':[np.nan]}
    
         # save for undo, then advance
         self.artists_stack.append(drawn)
         self._advance_prompt()
    
    def _advance_prompt(self):
        """increment index, update the onscreen prompt, redraw."""
        self.index += 1
        if self.index < len(self.labels):
            self.text.set_text(f"Click peak for: {self.labels[self.index]}")
        else:
            self.text.set_text("All peaks selected. Click 'Finished' to proceed.")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = event.key.lower()
        if key in ('shift+delete','shift+del','shift+backspace') and self.index > 0:
            # undo
            self.index -= 1
            label = self.labels[self.index]
            self.processed_data.pop(label, None)
            last = self.artists_stack.pop()
            for art in last:
                try: art.remove()
                except: pass
            self.text.set_text(f"Click peak for: {self.labels[self.index]}")
            self.fig.canvas.draw()

    def finish(self, event=None):
        # disconnect callbacks and close GUI
        self.text.set_text("")
        self.fig.canvas.draw
        self.fig.canvas.mpl_disconnect(self.cid_click)
        self.fig.canvas.mpl_disconnect(self.cid_key)
        self.finished=True
        QApplication.quit()