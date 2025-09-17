# src/chromatopy/chromatoPy_base.py
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pybaselines import Baseline
import warnings
warnings.simplefilter("always")
class GDGTAnalyzer:
    def __init__(self, df, traces, window_bounds, GDGT_dict, gaus_iterations, sample_name, is_reference, max_peaks, sw, sf, pk_sns, pk_pr, max_PA, reference_peaks=None, cheers=False, debug=False):
        self.fig, self.axs = None, None
        self.df = df
        self.traces = traces
        self.window_bounds = window_bounds
        self.GDGT_dict = GDGT_dict
        self.sample_name = sample_name
        self.is_reference = is_reference
        self.reference_peaks = reference_peaks  # ref_key
        self.fig, self.axs = None, None
        self.datasets = []
        self.peaks_indices = []
        self.integrated_peaks = {}
        self.action_stack = []
        self.no_peak_lines = {}
        self._nopeak_id = 0
        self.peaks = {}  # Store all peak indices and properties for each trace
        self.axs_to_traces = {}  # Empty map for connecting traces to figure axes
        self.peak_results = {}
        self.peak_results['Sample ID'] = sample_name
        self.gi = gaus_iterations
        self.max_peaks_for_neighborhood = max_peaks
        self.peak_properties = {}
        self.smoothing_params = [sw, sf]
        self.pk_sns = pk_sns
        self.pk_pr = pk_pr
        self.t_pressed = False # Flag to track if 't' was pressed
        self.called = False
        self.max_peak_amp = max_PA
        self.debug = debug
        self.cheers = cheers
        

    def run(self):
        """
        Executes the peak analysis workflow.
        Returns:
            peaks (dict): Peak areas and related info.
            fig (matplotlib.figure.Figure): The figure object.
            reference_peaks (dict): Updated reference peaks.
            t_pressed (bool): Indicates if 't' was pressed to update reference peaks.
        """
        self.fig, self.axs = self.plot_data()
        self.current_ax_idx = 0  # Initialize current axis index
        if self.is_reference:
            # Reference samples handling
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)  # Connect general key events
            plt.show(block=True)  # Blocks script until plot window is closed
            if not self.reference_peaks:
                self.reference_peaks = self.peak_results
            else:
                self.reference_peaks.update(self.peak_results)
        else:
            # Non-reference samples handling
            self.auto_select_peaks()
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            plt.show(block=True)  # Blocks script until plot window is closed
        return self.peak_results, self.fig, self.reference_peaks, self.t_pressed

    ######################################################
    ################  Gaussian Fit  ######################
    ######################################################
    def multigaussian(self, x, *params):
        """
        Computes the sum of multiple Gaussian functions for the given x-values and parameters.
        Parameters
        ----------
        x : numpy.ndarray
            Array of x-values where the Gaussian functions will be evaluated.
        *params : tuple of floats
            Variable-length argument list containing parameters for the Gaussian functions.
            Every three consecutive values represent the amplitude, center, and width
            of a Gaussian, in that order (amp, cen, wid).
        Returns
        -------
        y : numpy.ndarray
            The sum of all Gaussian functions evaluated at each x-value.
        """
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp = params[i]
            cen = params[i + 1]
            wid = params[i + 2]
            y += amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))
        return y

    # def gaussian_decay(self, x, amp, cen, wid, dec):
    #     """
    #     Computes a Gaussian function with an added exponential decay term.
    #     Parameters
    #     ----------
    #     x : numpy.ndarray
    #         Array of x-values where the Gaussian function will be evaluated.
    #     amp : float
    #         Amplitude of the Gaussian function (peak height).
    #     cen : float
    #         Center of the Gaussian function (peak position).
    #     wid : float
    #         Width of the Gaussian function (standard deviation of the distribution).
    #     dec : float
    #         Decay factor applied to the Gaussian to introduce exponential decay.
    #     Returns
    #     -------
    #     numpy.ndarray
    #         The values of the Gaussian function with exponential decay evaluated at each x-value.
    #     """
    #     return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2)) * np.exp(-dec * abs(x - cen))

    def individual_gaussian(self, x, amp, cen, wid):
        """
        Computes a single Gaussian function for the given x-values and parameters.
        Parameters
        ----------
        x : numpy.ndarray
            Array of x-values where the Gaussian function will be evaluated.
        amp : float
            Amplitude of the Gaussian function (peak height).
        cen : float
            Center of the Gaussian function (peak position).
        wid : float
            Width of the Gaussian function (standard deviation of the distribution).

        Returns
        -------
        numpy.ndarray
            The values of the Gaussian function evaluated at each x-value.
        """
        return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))

    def estimate_initial_gaussian_params(self, x, y, peak):
        """
        Estimates initial parameters for a Gaussian function, including height, mean, and standard deviation,
        based on the given x and y data and the specified peak.

        Parameters
        ----------
        x : pandas.Series or numpy.ndarray
            Array or series of x-values (typically the independent variable, e.g., time or retention time).
        y : pandas.Series or numpy.ndarray
            Array or series of y-values (typically the dependent variable, e.g., intensity or absorbance).
        peak : int
            Index of the peak in the x and y data around which to estimate the Gaussian parameters.

        Returns
        -------
        heights : list of float
            Estimated heights (amplitudes) of the Gaussian peaks.
        means : list of float
            Estimated means (centers) of the Gaussian peaks.
        stddevs : list of float
            Estimated standard deviations (widths) of the Gaussian peaks.

        Notes
        -----
        - The height is taken as the y-value at the peak index.
        - The mean is the x-value at the peak index.
        - The standard deviation is estimated from the full width at half maximum (FWHM) of the peak, or a rough estimate if the data is insufficient.
        """
        # Subset peaks so that only idx positions with x bounds are considered
        heights = []
        means = []
        stddevs = []

        height = y[peak]
        mean = x[peak]
        half_max = 0.5 * height

        mask = y >= half_max
        valid_x = x[mask]
        if len(valid_x) > 1:
            fwhm = np.abs(valid_x.iloc[-1] - valid_x.iloc[0])
            stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            stddev = (x.max() - x.min()) / 6
        heights.append(height)
        means.append(mean)
        stddevs.append(stddev)
        return heights, means, stddevs

    ######################################################
    ###############  Peak detection  #####################
    ######################################################

    def find_valleys(self, y, peaks, peak_oi=None):
        """
        Identifies valleys (lowest points) between peaks in the given data.

        Parameters
        ----------
        y : numpy.ndarray or pandas.Series
            Array or series of y-values (e.g., intensity or absorbance) from which valleys will be identified.
        peaks : numpy.ndarray or list of int
            List of indices representing the positions of the peaks in the data.
        peak_oi : int, optional
            Specific peak of interest. If provided, valleys adjacent to this peak will be identified;
            otherwise, valleys between all consecutive peaks will be identified.

        Returns
        -------
        valleys : list of int
            List of indices representing the positions of the valleys in the data.

        Notes
        -----
        - If `peak_oi` is None, the function finds valleys between all consecutive peaks in the dataset.
        - If `peak_oi` is provided, the function finds only the valleys surrounding the specified peak of interest.
        - Valleys are identified as the points of lowest y-values between consecutive peaks.
        """
        valleys = []
        if peak_oi == None:
            for i in range(1, len(peaks)):
                valley_point = np.argmin(y[peaks[i - 1] : peaks[i]]) + peaks[i - 1]
                valleys.append(valley_point)
        else:
            poi = np.where(peaks == peak_oi)[0][0]
            valleys.append(np.argmin(y[peaks[poi - 1] : peaks[poi]]) + peaks[poi - 1])
            valleys.append(np.argmin(y[peaks[poi] : peaks[poi + 1]]) + peaks[poi])
        return valleys
    
    # def _dbg(self, **kv):
    #     if getattr(self, "debug", False):
    #         print("[chromatoPy DBG]", {k: (float(v) if np.isscalar(v) else v) for k, v in kv.items()})
            
    # def find_peak_neighborhood_boundaries(self, x, y_smooth, peaks, valleys,
    #                                       peak_idx, ax, max_peaks, trace):
    #     """
    #     Find the extended neighborhood of peak_idx by fitting nearby peaks,
    #     extending each as a Gaussian, then taking the union of intervals that
    #     overlap the POI interval (symmetric interval intersection).
    #     """
    #     assert len(x) == len(y_smooth), f"Length mismatch: x={len(x)} y={len(y_smooth)}"
    #     assert np.all(np.isfinite(x)), "x has non-finite"
    #     assert np.all(np.diff(x) >= 0), "x must be sorted/non-decreasing"
    #     x_arr = np.asarray(x)
    #     y_arr = np.asarray(y_smooth)
    
    #     # Find closest peaks
    #     dists = np.abs(x_arr[peaks] - x_arr[peak_idx])
    #     closest_peaks_indices = np.argsort(dists)[:max_peaks]
    #     closest_peaks = np.sort(peaks[closest_peaks_indices])
    
    #     extended_boundaries = {}
    
    #     # ---- Fit and extend each candidate peak ----
    #     print(closest_peaks)
    #     for p in closest_peaks:
    #         # Map to properties index
    #         try:
    #             ppos = int(np.where(peaks == p)[0][0])
    #         except Exception:
    #             continue
    
    #         # Left/right base indices from scipy.find_peaks properties
    #         try:
    #             l_lim = int(self.peak_properties[trace]["left_bases"][ppos])
    #             r_lim = int(self.peak_properties[trace]["right_bases"][ppos])
    #         except Exception:
    #             # Fallback to a small window around the peak
    #             l_lim = max(p - 5, 0)
    #             r_lim = min(p + 5, len(x_arr) - 1)
    
    #         # Windowed data for a more stable single-Gaussian fit
    #         xw = x_arr[l_lim:r_lim + 1]
    #         yw = y_arr[l_lim:r_lim + 1]
    #         if xw.size < 3 or np.all(yw == 0):
    #             continue
    
    #         # Initial guess from local window
    #         try:
    #             heights, means, stddevs = self.estimate_initial_gaussian_params(
    #                 pd.Series(xw), pd.Series(yw), int(np.argmin(np.abs(xw - x_arr[p]))))
    #             height, mean, stddev = heights[0], means[0], stddevs[0]
    #             stddev = max(float(stddev), 1e-6)
    #         except Exception:
    #             # Crude fallback
    #             height = float(y_arr[p])
    #             mean = float(x_arr[p])
    #             stddev = max((xw.max() - xw.min()) / 6.0, 1e-6)
    
    #         # Fit a single Gaussian to the local window
    #         try:
    #             popt, _ = curve_fit(self.individual_gaussian, xw, yw,
    #                                 p0=[height, mean, stddev],
    #                                 maxfev=self.gi)
    #         except Exception:
    #             # Retry with looser maxfev; if it fails, skip this peak
    #             try:
    #                 popt, _ = curve_fit(self.individual_gaussian, xw, yw,
    #                                     p0=[height, mean, stddev],
    #                                     maxfev=self.gi * 10)
    #             except Exception:
    #                 continue
    
    #         amp, cen, wid = float(popt[0]), float(popt[1]), max(float(popt[2]), 1e-6)
    
    #         # Extend this Gaussian to get an "influence interval"
    #         x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, decay=0.0, factor=1.0)
    #         ext_x, ext_y = self.extrapolate_gaussian(xw, amp, cen, wid, None, x_min, x_max, step=0.01)
            
    #         self._dbg(peak_index=int(p), cen=cen, wid=wid, x_min=x_min, x_max=x_max, step=0.01, ext_n=int(len(ext_x)))
    #         dx_local = np.median(np.diff(xw)) if len(xw) > 1 else np.nan 
    #         self._dbg(l_lim=l_lim, r_lim=r_lim, xw_n=int(len(xw)), dx_local=dx_local, amp=amp)
    
    #         # Find left/right boundaries on the extended curve (derivative-based)
    #         try:
    #             # index of this peak center in ext_x
    #             n_idx = int(np.argmin(np.abs(ext_x - cen)))
    #             L, R = self.calculate_boundaries(ext_x, ext_y, n_idx)
    #             # Guard / normalize
    #             L = max(int(L), 0)
    #             R = min(int(R), len(ext_x) - 1)
    #             if L >= R:
    #                 # Degenerate; fallback to a small interval around center
    #                 hw = max(int(round(5)), 1)
    #                 L = max(n_idx - hw, 0)
    #                 R = min(n_idx + hw, len(ext_x) - 1)
    #             extended_boundaries[p] = (float(ext_x[L]), float(ext_x[R]))
    #         except Exception:
    #             # Fallback interval if derivative boundaries fail
    #             hw = max((x_max - x_min) * 0.05, np.median(np.diff(ext_x)) * 5.0)
    #             extended_boundaries[p] = (float(cen - hw), float(cen + hw))
    
    #     # ---- Ensure POI (peak of interest) has bounds ----
    #     poi_left, poi_right = extended_boundaries.get(peak_idx, (None, None))
    #     if poi_left is None or poi_right is None:
    #         # Build a conservative fallback around the POI center
    #         cen = float(x_arr[peak_idx])
    #         # Half-width fallback ~ 5 samples (or 0.005 min if spacing is extremely fine)
    #         dx = np.median(np.diff(x_arr)) if len(x_arr) > 1 else 0.01
    #         hw = max(5 * dx, 1e-3)
    #         poi_left, poi_right = cen - hw, cen + hw
    #         extended_boundaries[peak_idx] = (poi_left, poi_right)
    
    #     # ---- Symmetric interval overlap test (no left/right bias) ----
    #     def intervals_overlap(aL, aR, bL, bR, eps=0.0):
    #         if aL > aR: aL, aR = aR, aL
    #         if bL > bR: bL, bR = bR, bL
    #         return (aL <= bR - eps) and (bL <= aR - eps)
    
    #     overlapping_peaks = []
    #     for p, (L, R) in extended_boundaries.items():
    #         if p == peak_idx:
    #             continue
    #         if L is None or R is None:
    #             continue
    #         if intervals_overlap(L, R, poi_left, poi_right, eps=0.0):
    #             overlapping_peaks.append(p)
    
    #     # ---- Neighborhood bounds from overlaps (or POI-only if none) ----
    #     if overlapping_peaks:
    #         left_most_peak = min(overlapping_peaks, key=lambda q: extended_boundaries[q][0])
    #         right_most_peak = max(overlapping_peaks, key=lambda q: extended_boundaries[q][1])
    #         neighborhood_left_boundary = extended_boundaries[left_most_peak][0]
    #         neighborhood_right_boundary = extended_boundaries[right_most_peak][1]
    #     else:
    #         neighborhood_left_boundary, neighborhood_right_boundary = poi_left, poi_right
    
    #     return neighborhood_left_boundary, neighborhood_right_boundary, overlapping_peaks
    
    def _safe_for_json(self, v):
        # numeric scalars → float; numpy scalars → float; everything else → str if not JSONable
        try:
            if isinstance(v, (int, float, np.number)):
                return float(v)
            return v
        except Exception:
            return str(v)
    
    def _fmt(self, kv):
        return "[chromatoPy DBG] " + json.dumps({k: self._safe_for_json(v) for k, v in kv.items()},
                                                default=str)
    
    def _dbg(self, **kv):
        """Verbose debug (only prints if self.debug=True). Never throws."""
        if self.debug:
            try:
                warnings.warn(self._fmt(kv), stacklevel=2)
            except Exception:
                # last-ditch: never let logging crash the app
                warnings.warn("[chromatoPy DBG] <logging failed>", stacklevel=2)
    
    def _err(self, **kv):
        """ALWAYS print (even if self.debug=False). Use for exceptions/fallbacks."""
        try:
            warnings.warn(self._fmt(dict(level="ERROR", **kv)), stacklevel=2)
        except Exception:
            warnings.warn("[chromatoPy DBG] <logging failed>", stacklevel=2)


    def find_peak_neighborhood_boundaries(self, x, y_smooth, peaks, valleys,
                                          peak_idx, ax, max_peaks, trace):
        """
        Fit nearby peaks with single Gaussians, extend each to an 'influence' interval,
        take union of intervals overlapping the peak-of-interest (POI).
        Returns (neighborhood_left_boundary, neighborhood_right_boundary, overlapping_peaks)
        """
        # Preconditions
        assert len(x) == len(y_smooth), f"Length mismatch: x={len(x)} y={len(y_smooth)}"
        assert np.all(np.isfinite(x)), "x has non-finite"
        assert np.all(np.diff(x) >= 0), "x must be sorted/non-decreasing"
    
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y_smooth, dtype=float)
    
        # Ensure peak_idx is an index into x (not into peaks array)
        if 0 <= peak_idx < len(peaks) and (0 <= int(peaks[peak_idx]) < len(x_arr)):
            peak_idx = int(peaks[peak_idx])
            self._dbg(peak_idx_remapped=True, new_peak_idx=peak_idx)
    
        peak_idx = int(np.clip(peak_idx, 0, len(x_arr) - 1))
    
        # Find closest peaks by distance in x
        dists = np.abs(x_arr[peaks] - x_arr[peak_idx])
        closest_peaks_indices = np.argsort(dists)[:max_peaks]
        closest_peaks = np.sort(peaks[closest_peaks_indices])
    
        extended_boundaries = {}
    
        # ---- Fit and extend each candidate peak ----
        for p in closest_peaks:
            try:
                ppos = int(np.where(peaks == p)[0][0])
            except Exception as e:
                self._dbg(skip_peak_no_ppos=True, p=int(p), exc=str(e))
                continue
    
            # Initial bases from find_peaks
            try:
                l_lim = int(self.peak_properties[trace]["left_bases"][ppos])
                r_lim = int(self.peak_properties[trace]["right_bases"][ppos])
            except Exception as e:
                l_lim = max(int(p) - 5, 0)
                r_lim = min(int(p) + 5, len(x_arr) - 1)
                self._dbg(bases_missing=True, p=int(p), l_lim=l_lim, r_lim=r_lim, exc=str(e))
            # --- Clamp window by valleys and width ---
            dx = np.median(np.diff(x_arr)) if len(x_arr) > 1 else 0.01
            v_left = max([v for v in valleys if v < p], default=l_lim)
            v_right = min([v for v in valleys if v > p], default=r_lim)
            l_lim = max(l_lim, v_left)
            r_lim = min(r_lim, v_right)
    
            # crude sigma estimate (fallback to a few points)
            sig_est = max((x_arr[r_lim] - x_arr[l_lim]) / 20.0, 3 * dx)
            max_span = int(np.ceil(6 * sig_est / dx))
            l_lim = max(p - max_span, l_lim, 0)
            r_lim = min(p + max_span, r_lim, len(x_arr) - 1)
    
            # Hard guard: cap window size
            if (r_lim - l_lim) > 1000:
                self._dbg(clamp_huge_window=True, p=int(p), span=r_lim-l_lim)
                half = 200
                l_lim = max(p - half, 0)
                r_lim = min(p + half, len(x_arr)-1)    
            xw = x_arr[l_lim:r_lim+1]
            yw = y_arr[l_lim:r_lim+1]
            if xw.size < 3 or np.all(yw == 0):
                self._dbg(skip_peak_tiny_or_flat=True, p=int(p), xw_n=int(xw.size))
                continue
    
            # --- Initial guess ---
            try:
                heights, means, stddevs = self.estimate_initial_gaussian_params(
                    pd.Series(xw), pd.Series(yw), int(np.argmin(np.abs(xw - x_arr[p]))))
                height, mean, stddev = heights[0], means[0], max(float(stddevs[0]), 1e-6)
            except Exception as e:
                height = float(y_arr[p])
                mean = float(x_arr[p])
                stddev = max((xw.max() - xw.min()) / 6.0, 1e-6)
                self._dbg(init_guess_fallback=True, p=int(p), exc=str(e),
                          height=height, mean=mean, stddev=stddev)
    
            # --- Fit Gaussian with bounds ---
            lb = [0.0, xw.min(), dx]  # amp>=0, center>=min, width>=dx
            ub = [np.inf, xw.max(), (xw.max()-xw.min())/2.0]
            try:
                popt, _ = curve_fit(self.individual_gaussian, xw, yw,
                                    p0=[height, mean, stddev],
                                    bounds=(lb, ub),
                                    maxfev=self.gi)
            except Exception as e1:
                try:
                    popt, _ = curve_fit(self.individual_gaussian, xw, yw,
                                        p0=[height, mean, stddev],
                                        bounds=(lb, ub),
                                        maxfev=self.gi*5)
                    self._dbg(refit_success=True, p=int(p))
                except Exception as e2:
                    self._err(skip_peak_fit_fail=True, p=int(p),
                              e1=str(e1), e2=str(e2), xw_n=len(xw))
                    continue
    
            amp, cen, wid = float(popt[0]), float(popt[1]), max(float(popt[2]), 1e-6)
    
            # --- Extend Gaussian ---
            x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, decay=0.0, factor=1.0)
            ext_x, ext_y = self.extrapolate_gaussian(xw, amp, cen, wid, None, x_min, x_max, step=0.01)
    
            self._dbg(peak_index=int(p), cen=cen, wid=wid,
                      x_min=x_min, x_max=x_max, ext_n=len(ext_x))
    
            # --- Boundaries ---
            try:
                n_idx = int(np.argmin(np.abs(ext_x - cen)))
                L, R = self.calculate_boundaries(ext_x, ext_y, n_idx)
                L = max(int(L), 0)
                R = min(int(R), len(ext_x) - 1)
                if L >= R:  # fallback
                    hw = max(5, 1)
                    L = max(n_idx - hw, 0)
                    R = min(n_idx + hw, len(ext_x)-1)
                extended_boundaries[p] = (float(ext_x[L]), float(ext_x[R]))
            except Exception as e:
                self._err(err="calc_boundaries_failed", p=int(p), exc=str(e),
                          ext_n=len(ext_x), cen=cen, wid=wid)
                dx_ext = np.median(np.diff(ext_x)) if len(ext_x) > 1 else dx
                hw = max((x_max - x_min)*0.05, dx_ext*5.0)
                extended_boundaries[p] = (float(cen - hw), float(cen + hw))
    
        # ---- Ensure POI has bounds ----
        poi_left, poi_right = extended_boundaries.get(peak_idx, (None, None))
        if poi_left is None or poi_right is None:
            cen = float(x_arr[peak_idx])
            dx = np.median(np.diff(x_arr)) if len(x_arr) > 1 else 0.01
            hw = max(5 * dx, 1e-3)
            poi_left, poi_right = cen - hw, cen + hw
            extended_boundaries[peak_idx] = (poi_left, poi_right)
            self._dbg(poi_fallback=True, cen=cen, hw=hw)
    
        # ---- Overlap check ----
        def intervals_overlap(aL, aR, bL, bR, eps=0.0):
            if aL > aR: aL, aR = aR, aL
            if bL > bR: bL, bR = bR, aL
            return (aL <= bR - eps) and (bL <= aR - eps)
    
        overlapping_peaks = []
        for p, (L, R) in extended_boundaries.items():
            if p == peak_idx: continue
            if L is None or R is None: continue
            if intervals_overlap(L, R, poi_left, poi_right):
                overlapping_peaks.append(p)
    
        # ---- Final neighborhood ----
        if overlapping_peaks:
            left_most = min(overlapping_peaks, key=lambda q: extended_boundaries[q][0])
            right_most = max(overlapping_peaks, key=lambda q: extended_boundaries[q][1])
            neighborhood_left_boundary = extended_boundaries[left_most][0]
            neighborhood_right_boundary = extended_boundaries[right_most][1]
        else:
            neighborhood_left_boundary, neighborhood_right_boundary = poi_left, poi_right
        self._dbg(neighborhood=True, poi=(poi_left, poi_right),
                  bounds=(neighborhood_left_boundary, neighborhood_right_boundary),
                  overlaps=[int(v) for v in overlapping_peaks])
        return neighborhood_left_boundary, neighborhood_right_boundary, overlapping_peaks
    
    
    def calculate_boundaries(self, x, y, ind_peak, tolerance=0.02, w_factor=3.0):
        """
        Minimal, *local* derivative method:
          - Smooth y, compute smoothed derivative.
          - Build a local window around the apex from half-maximum width.
          - Find |slope| peak (shoulder) on each side within that window.
          - Walk outward until |slope| <= tolerance * |shoulder_slope|.
        """
        import numpy as np
    
        x = np.asarray(x, float)
        y = np.asarray(y, float)
    
        y_s = self.smoother(y)
        vel, _ = self.forward_derivative(x, y_s)
        # print(f"x\n{x}")
        # print(f"y\n{y}")
        # print(f"vel\n{vel}")
        vel = np.asarray(vel, float)
        if vel.size == x.size - 1 and vel.size > 0:
            vel = np.r_[vel, vel[-1]]
    
        # basic sanitation + light smoothing for stability
        vel[~np.isfinite(vel)] = 0.0
        vel_s = self.smoother(vel)
        abs_vel = np.abs(vel_s)
    
        N = y_s.size
        p = int(np.clip(ind_peak, 0, N - 1))
    
        # --- local window from half-maximum ---
        peak_y = y_s[p]
        half = 0.5 * peak_y
        L = p
        while L > 0 and y_s[L] > half:
            L -= 1
        R = p
        while R < N - 1 and y_s[R] > half:
            R += 1
        # widen a bit to comfortably include shoulders
        W = max(5, int((R - L) * w_factor / 2))
        L0 = max(0, p - W)
        R0 = min(N - 1, p + W)
    
        # --- shoulders via |slope| peaks within the local halves ---
        if p > L0:
            left_shoulder = L0 + int(np.argmax(abs_vel[L0:p]))
        else:
            left_shoulder = p
        if R0 > p + 1:
            right_shoulder = (p + 1) + int(np.argmax(abs_vel[p+1:R0+1]))
        else:
            right_shoulder = p
    
        # thresholds from shoulder strength (avoid zero by epsilon)
        eps = 1e-12
        # print(vel_s)
        left_thr  = max(tolerance * abs(vel_s[left_shoulder]), eps)
        right_thr = max(tolerance * abs(vel_s[right_shoulder]), eps)
    
        # --- walk into tails, but stay inside the local window ---
        A = left_shoulder
        while A > L0 and abs_vel[A] > left_thr:
            A -= 1
    
        B = right_shoulder
        while B < R0 and abs_vel[B] > right_thr:
            B += 1
    
        # ensure order (rare, but cheap)
        if A >= B:
            A = max(L0, p - 3)
            B = min(R0, p + 3)
    
        return int(A), int(B)

    def find_peak_boundaries(self, x, y, center, trace, threshold=0.1):
        """
        Finds the left and right boundaries of a peak based on the first derivative test and a threshold value.

        Parameters
        ----------
        x : numpy.ndarray or pandas.Series
            Array of x-values (e.g., retention times) corresponding to the data points.
        y : numpy.ndarray or pandas.Series
            Array of y-values (e.g., signal intensities) corresponding to the x-values.
        center : float
            The x-value around which the peak is centered (peak of interest).
        trace : str
            Identifier for the trace being analyzed (e.g., which sample or dataset the peak belongs to).
        threshold : float, optional
            Threshold value for the derivative used to determine the boundaries. Default is 0.1.

        Returns
        -------
        left_boundary_index : int
            Index of the left boundary of the peak.
        right_boundary_index : int
            Index of the right boundary of the peak.

        Notes
        -----
        - The function calculates the first derivative of the y-values with respect to the x-values to detect changes in slope.
        - The boundaries are determined by finding where the derivative falls below a threshold before and after the peak center.
        - If no suitable left boundary is found, the function defaults to the start of the x-array.
        - If no suitable right boundary is found, the function defaults to the end of the x-array.
        - The small epsilon value is added to the denominator to avoid division by zero during derivative calculation.
        """

        # Reset index and keep the original index as a column
        new_ind_peak = min(range(len(x)), key=lambda i: abs(x[i] - center))
        dx = np.diff(x)
        dy = np.abs(np.diff(y))
        epsilon = 1e-10
        derivative = dy / (dx + epsilon)

        # Normalize the derivative for stability in thresholding
        derivative /= np.max(np.abs(derivative))
        # Search for the last point where derivative is greater than threshold before the peak
        left_candidates = np.where((np.abs(derivative[:new_ind_peak]) < threshold))[0]  # | (derivative[:new_ind_peak] < -threshold))[0]
        if left_candidates.size > 0:
            left_boundary_index = left_candidates[-1]
        else:
            left_boundary_index = 0  # Start of the array if no suitable point is found

        # Search for the first point where derivative is greater than threshold after the peak
        right_candidates = np.where((derivative[new_ind_peak:] < threshold))[0]  # np.where((derivative[new_ind_peak:] > threshold) | (derivative[new_ind_peak:] < -threshold))[0]
        if right_candidates.size > 0:
            right_boundary_index = right_candidates[0] + new_ind_peak
        else:
            right_boundary_index = len(x) - 1  # End of the array if no suitable point is found
        return int(left_boundary_index), int(right_boundary_index)

    def smoother(self, y, param_0 = None, param_1 = None):
        """
        Applies a Savitzky-Golay filter to smooth the given data.

        Parameters
        ----------
        y : numpy.ndarray or pandas.Series
            Array or series of y-values (e.g., signal intensities) that will be smoothed.

        Returns
        -------
        numpy.ndarray
            The smoothed y-values after applying the Savitzky-Golay filter.

        Notes
        -----
        - This function uses the `savgol_filter` from `scipy.signal`, which applies a Savitzky-Golay filter to smooth the data.
        - The smoothing parameters, such as the window length and polynomial order, are stored in `self.smoothing_params`.
            - `self.smoothing_params[0]`: Window length (must be odd).
            - `self.smoothing_params[1]`: Polynomial order for the filter.
        """
        if param_0 == None:
            param_0 = self.smoothing_params[0]
        if param_1 == None:
            param_1 = self.smoothing_params[1]
        # return savgol_filter(y, self.smoothing_params[0], self.smoothing_params[1], deriv=0, mode='interp')
        if len(y) > param_0:
            return savgol_filter(y, param_0, param_1)
        else:
            return y

    def forward_derivative(self, x, y):
        """
        Computes the forward first derivative of the y-values with respect to the x-values.

        Parameters
        ----------
        x : numpy.ndarray or pandas.Series
            Array or series of x-values (e.g., time, retention time) corresponding to the data points.
        y : numpy.ndarray or pandas.Series
            Array or series of y-values (e.g., signal intensities) corresponding to the x-values.

        Returns
        -------
        fd : numpy.ndarray
            The first derivative of the y-values with respect to the x-values (forward difference).
        x_n : numpy.ndarray
            The x-values corresponding to the first derivative, excluding the last element of the original x array.

        Notes
        -----
        - The forward difference method is used to calculate the derivative, which approximates the slope between consecutive points.
        - The derivative array `FD1` will have one less element than the original y-values due to the nature of finite differences.
        - `x_n` excludes the last element of `x` to match the size of `fd`.
        """
        fd = np.diff(y) / np.diff(x)
        x_n = x[:-1]
        return fd, x_n

    def extrapolate_gaussian(self, x, amp, cen, wid, decay, x_min, x_max, step=0.001): # Modified step from 0.01 17092025
        """
        Extends the Gaussian function by extrapolating its tails between x_min and x_max with a specified step size.

        Parameters
        ----------
        x : numpy.ndarray or pandas.Series
            Array of x-values (e.g., time or retention time) where the original Gaussian data is located.
        amp : float
            Amplitude of the Gaussian function (peak height).
        cen : float
            Center of the Gaussian function (peak position).
        wid : float
            Width of the Gaussian function (standard deviation of the distribution).
        decay : float or None
            Decay factor applied to the Gaussian function to introduce exponential decay. If `None`, no decay is applied.
        x_min : float
            The minimum x-value for the extrapolation range.
        x_max : float
            The maximum x-value for the extrapolation range.
        step : float, optional
            Step size for generating new x-values between x_min and x_max. Default is 0.1.

        Returns
        -------
        extended_x : numpy.ndarray
            Array of x-values extended from x_min to x_max with the given step size.
        extended_y : numpy.ndarray
            Array of y-values corresponding to the extrapolated Gaussian function over the extended x-values.

        Notes
        -----
        - If `decay` is `None`, the function will apply a simple Gaussian using `self.individual_gaussian`.
        - If `decay` is provided, it applies a Gaussian function with exponential decay using `self.gaussian`.
        """
        extended_x = np.arange(x_min, x_max, step)
        if decay is None:
            extended_y = self.individual_gaussian(extended_x, amp, cen, wid)
        else:
            extended_y = self.gaussian_decay(extended_x, amp, cen, wid, decay)
        return extended_x, extended_y

    def calculate_gaus_extension_limits(self, cen, wid, decay=None, factor=2.0, max_tail_sigma=3.0):
        """
        Compute extension limits for a Gaussian-like peak. If `decay` is None or invalid,
        fall back to a symmetric sigma-capped tail.
    
        Parameters
        ----------
        cen : float         # center
        wid : float         # sigma (or whatever you use consistently here)
        decay : float|None  # positive => use 1/decay tail; None/<=0 => cap to max_tail_sigma*sigma
        factor : float      # how many sigmas to include before tail
        max_tail_sigma : float
            Max extra tail length in units of sigma (safety cap).
    
        Returns
        -------
        x_min, x_max : float
        """
        sigma = wid * factor
        tail_cap = sigma * max_tail_sigma
    
        use_decay = (decay is not None) and np.isfinite(decay) and (decay > 0)
        tail = min(1.0 / decay, tail_cap) if use_decay else tail_cap
    
        x_min = cen - sigma - tail
        x_max = cen + sigma + tail
    
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            pad = max(1e-6, 0.01 * max(abs(sigma), 1.0))
            x_min, x_max = cen - pad, cen + pad
        return x_min, x_max
        
    # def fit_gaussians(self, x_full, y_full, ind_peak, trace, peaks, ax):
    #     """
    #     Fits single or multi-Gaussian models to the provided data to determine the best-fit parameters for the peaks of interest.

    #     Parameters
    #     ----------
    #     x_full : numpy.ndarray or pandas.Series
    #         Array of full x-values (e.g., retention times) corresponding to the data points.
    #     y_full : numpy.ndarray or pandas.Series
    #         Array of full y-values (e.g., signal intensities) corresponding to the x-values.
    #     ind_peak : int
    #         Index of the peak of interest in the data.
    #     trace : str
    #         Identifier for the trace being analyzed (e.g., which sample or dataset the peak belongs to).
    #     peaks : list of int
    #         List of indices representing the detected peaks in the data.
    #     ax : matplotlib.axes.Axes
    #         The axis object used for plotting the Gaussian fits.

    #     Returns
    #     -------
    #     best_x : numpy.ndarray
    #         Array of x-values corresponding to the best Gaussian fit (single or multi-Gaussian) for the peak of interest.
    #     best_fit_y : numpy.ndarray
    #         Array of y-values corresponding to the best Gaussian fit (single or multi-Gaussian) for the peak of interest.
    #     area_smooth : float
    #         The area under the curve for the best-fit Gaussian model, calculated using Simpson's rule.

    #     Notes
    #     -----
    #     - The function iteratively fits multi-Gaussian models to detect overlapping peaks and determine the best fit.
    #     - If a multi-Gaussian model does not provide a satisfactory fit, the function tries to fit a single Gaussian with exponential decay.
    #     - Gaussian parameters such as amplitude, center, and width are estimated using initial guesses and bounded constraints.
    #     - The function calculates boundaries for peak fitting based on the first derivative and extends the Gaussian tails beyond the peak region.
    #     - The best fit is determined based on the lowest root mean square error (RMSE) between the fitted Gaussian and the observed data.
    #     - The function returns the best-fit x and y values, along with the area under the curve using Simpson's rule for numerical integration.
    #     """
    #     # detect overlapping peaks
    #     current_peaks = np.array(peaks)
    #     current_peaks = np.append(current_peaks, ind_peak)
    #     current_peaks = np.sort(current_peaks)
    #     iteration = 0
    #     best_fit_y = None
    #     best_x = None
    #     best_fit_params = None
    #     best_ksp = np.inf
    #     multi_gauss_flag = True
    #     best_idx_interest = None
    #     best_error = np.inf
    #     best_ks_stat = np.inf
    #     while len(current_peaks) > 1:
    #         left_boundary, _ = self.calculate_boundaries(x_full, y_full, np.min(current_peaks))
    #         _, right_boundary = self.calculate_boundaries(x_full, y_full, np.max(current_peaks))
    #         x = x_full[left_boundary : right_boundary + 1]
    #         y = y_full[left_boundary : right_boundary + 1]
    #         index_of_interest = np.where(current_peaks == ind_peak)[0][0]
    #         initial_guesses = []
    #         bounds_lower = []
    #         bounds_upper = []
    #         for peak in current_peaks:
    #             height, center, width = self.estimate_initial_gaussian_params(x, y, peak)  # 
    #             height = height[0]
    #             center = center[0]
    #             width = width[0]
    #             initial_guesses.extend([height, center, width])
    #             # Bounds for peak fitting
    #             lw = 0.1 - width if width > 0.1 else width
    #             bounds_lower.extend([0.1 * y_full[peak], x_full[peak] - 0.15, lw])  # Bounds for peak fittin
    #             bounds_upper.extend([1 + y_full[peak], x_full[peak] + 0.15, 0.5 + width])  # Old amplitude was 2 * peak height, y_full[peak] * 2, width was 2+width
    #         bounds = (bounds_lower, bounds_upper)
    #         try:
    #             # with warnings.catch_warnings():
    #             #     warnings.simplefilter("ignore")
    #                 # popt, pcov = curve_fit(self.multigaussian, x, y, p0=initial_guesses, method="dogbox", bounds=bounds, maxfev=self.gi)  # , ftol=1e-4, xtol=1e-4)
    #             popt, pcov = curve_fit(self.multigaussian, x, y, p0=initial_guesses, method="trf", bounds=bounds, maxfev=self.gi)  # , ftol=1e-4, xtol=1e-4) INDENT IF WE PUT BACK WARNINGS
    #             # popt, pcov = curve_fit(self.multigaussian, x, y, p0=initial_guesses, method="trf", maxfev=self.gi*100)
    #             fitted_y = self.multigaussian(x, *popt)
    #             ax.plot(x, fitted_y, c="fuchsia", alpha = 0.4) # plots the multi gaussian curve
    #             error = np.sqrt(((fitted_y - y) ** 2).mean())  # RMSE
    #             if error < best_error:
    #                 best_error = error
    #                 best_fit_params = popt
    #                 best_fit_params_error = pcov
    #                 best_fit_y = fitted_y
    #                 best_x = x
    #                 best_idx_interest = index_of_interest
    #         except RuntimeError:
    #             pass
    #         distances = np.abs(x[current_peaks] - x_full[ind_peak])
    #         if distances.size > 0:
    #             max_dist_idx = np.argmax(distances)
    #             current_peaks = np.delete(current_peaks, max_dist_idx)
    #         iteration += 1

    #     # Final fit with only the selected peak
    #     if len(current_peaks) == 1:
    #         left_boundary, right_boundary = self.calculate_boundaries(x_full, y_full, ind_peak)
    #         x = x_full[left_boundary : right_boundary + 1]
    #         y = y_full[left_boundary : right_boundary + 1]
    #         height, center, width = self.estimate_initial_gaussian_params(x, y, ind_peak)
    #         height = height[0]
    #         center = center[0]
    #         width = width[0]
    #         # p0 = [height, center, width]
    #         initial_decay = 0.1
    #         p0 = [height, center, width, initial_decay]
    #         # bounds_lower = [0.9 * y_full[ind_peak], x_full[ind_peak] - 0.1, 0.5 * width, 0.01]  # modified width from 0.05
    #         # bounds_upper = [1 + y_full[ind_peak], x_full[ind_peak] + 0.1, width * 1.5, 2]
    #         # bounds = (bounds_lower, bounds_upper)
    #         try:
    #             # Initial try with given maxfev
    #             # with warnings.catch_warnings():
    #             #     warnings.simplefilter("ignore")
    #                 # single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: self.gaussian_decay(x, amp, cen, wid, dec), x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=self.gi)
    #             # single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: self.gaussian_decay(x, amp, cen, wid, dec), x, y, p0=p0, method="trf", bounds=bounds, maxfev=self.gi)
    #             single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: self.gaussian_decay(x, amp, cen, wid, dec), x, y, p0=p0, method="trf",maxfev=self.gi)
    #             single_fitted_y = self.gaussian_decay(x, *single_popt)
    #             error = np.sqrt(((single_fitted_y - y) ** 2).mean())  # RMSE
    #             if error < best_error:
    #                 multi_gauss_flag = False
    #                 best_error = error
    #                 best_fit_params = single_popt
    #                 best_fit_params_error = single_pcov
    #                 best_fit_y = single_fitted_y
    #                 best_x = x
    #         except RuntimeError:
    #             print(f"Warning: Optimal parameters could not be found with {self.gi} iterations. Increasing iterations by a factor of 100. Please be patient.")

    #             # Increase maxfev by a factor of 10 and retry
    #             try:
    #                 # with warnings.catch_warnings():
    #                 #     warnings.simplefilter("ignore")
    #                     # single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: self.gaussian_decay(x, amp, cen, wid, dec), x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=self.gi* 1000) # comment out to speed up debug
    #                 single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: self.gaussian_decay(x, amp, cen, wid, dec), x, y, p0=p0, method="trf", bounds=bounds, maxfev=self.gi)
    #                 single_fitted_y = self.gaussian_decay(x, *single_popt)
    #                 error = np.sqrt(((single_fitted_y - y) ** 2).mean())  # RMSE)
    #                 if error < best_error:
    #                     multi_gauss_flag = False
    #                     best_error = error
    #                     best_fit_params = single_popt
    #                     best_fit_params_error = single_pcov
    #                     best_fit_y = single_fitted_y
    #                     best_x = x
    #             except RuntimeError:
    #                 print("Error: Optimal parameters could not be found even after increasing the iterations.")
    #     if multi_gauss_flag == True:
    #         print(f"Selected multi: {trace}")
    #         # Determine the index of the peak of interest in the multi-Gaussian fit
            
    #         amp, cen, wid = best_fit_params[best_idx_interest * 3], best_fit_params[best_idx_interest * 3 + 1], best_fit_params[best_idx_interest * 3 + 2]
    #         x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, decay=None)
    #         # best_fit_y = self.individual_gaussian(best_x, amp, cen, wid)
    #         best_x, best_fit_y = self.extrapolate_gaussian(best_x, amp, cen, wid, None, x_min, x_max)
    #         new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
    #         left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
    #         best_x = best_x[left_boundary - 1 : right_boundary + 1]
    #         best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
    #         area_smooth, area_ensemble = self.peak_area_distribution(best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ax, ind_peak, multi=True)

    #     else:
    #         print(f"Selected single: {trace}")
    #         amp, cen, wid, dec = best_fit_params[0], best_fit_params[1], best_fit_params[2], best_fit_params[3]
    #         x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, dec, factor=2)
    #         best_x, best_fit_y = self.extrapolate_gaussian(best_x, amp, cen, wid, dec, x_min, x_max)
    #         new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
    #         left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
    #         best_x = best_x[left_boundary - 1 : right_boundary + 1]
    #         best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
    #         area_smooth, area_ensemble = self.peak_area_distribution(best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ax, ind_peak, multi = False)

    #     return best_x, best_fit_y, area_smooth, area_ensemble
    
    
    
    
    # def fit_gaussians(self, x_full, y_full, ind_peak, trace, peaks, ax):
    #     """
    #     Fits single or multi-Gaussian models. Multi-Gaussian uses L2 data residuals
    #     + small regularization terms to discourage overly broad peaks.
    
    #     New knobs (optional as attributes on self):
    #       self.lambda_sigma : float  (default 1e-2)  # strength of width penalty
    #       self.lambda_area  : float  (default 0.0)   # strength of area (A*σ) penalty
    #       self.use_robust   : bool   (default False) # keep pure L2 by default
    #       self.f_scale      : float  (default 1.0)   # scale for robust loss if enabled
    #     """
    #     import numpy as np
    #     from scipy.optimize import least_squares, curve_fit
    #     from numpy.linalg import inv, LinAlgError
    
    #     # Small epsilon to avoid div-by-zero
    #     EPS = 1e-12
    
    #     # --- Helpers ---
    #     def multigauss(x, params):
    #         # params = [A1, mu1, sig1, A2, mu2, sig2, ...]
    #         y = np.zeros_like(x, dtype=float)
    #         k = len(params) // 3
    #         for i in range(k):
    #             A, mu, sig = params[3*i:3*i+3]
    #             sig = abs(sig) + EPS  # be safe
    #             z = (x - mu) / sig
    #             y += A * np.exp(-0.5 * z*z)
    #         return y
    
    #     def lsq_with_sigma_area_penalty(x, y, p0, bounds, A0, sig0,
    #                                     lambda_sigma=1e-2, lambda_area=0.0,
    #                                     robust=False, f_scale=1.0, max_nfev=5000):
    #         """
    #         Pure L2 on data + small penalties:
    #           r = [ (y - yhat),
    #                 sqrt(lambda_sigma) * log(sig/sig0),
    #                 sqrt(lambda_area)  * ((A*sig - A0*sig0)/(A0*sig0+EPS)) ]
    #         """
    #         x = np.asarray(x, float)
    #         y = np.asarray(y, float)
    #         p0 = np.asarray(p0, float)
    #         lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    
    #         k = len(p0) // 3
    #         sigma_idx = [3*i+2 for i in range(k)]
    #         amp_idx   = [3*i+0 for i in range(k)]
    
    #         # Sanity for refs
    #         A0 = np.asarray(A0, float)
    #         sig0 = np.asarray(sig0, float)
    #         sig0 = np.where(sig0 <= 0, 1.0, sig0)
    
    #         sqrt_lsig  = np.sqrt(max(0.0, float(lambda_sigma)))
    #         sqrt_larea = np.sqrt(max(0.0, float(lambda_area)))
    
    #         def residuals(p):
    #             yhat = multigauss(x, p)
    #             r_data = (y - yhat)
    
    #             # penalties
    #             r_list = [r_data]
    
    #             if sqrt_lsig > 0:
    #                 sigs = np.abs(p[sigma_idx]) + EPS
    #                 r_sigma = sqrt_lsig * np.log(sigs / sig0)
    #                 r_list.append(r_sigma)
    
    #             if sqrt_larea > 0:
    #                 As   = p[amp_idx]
    #                 sigs = np.abs(p[sigma_idx]) + EPS
    #                 area     = As * sigs
    #                 area_ref = A0 * sig0
    #                 r_area = sqrt_larea * ( (area - area_ref) / (area_ref + EPS) )
    #                 r_list.append(r_area)
    
    #             return np.concatenate(r_list, axis=0)
    
    #         res = least_squares(
    #             residuals, p0, bounds=(lo, hi),
    #             loss=('dogbox' if robust else 'linear'),
    #             f_scale=f_scale, max_nfev=max_nfev)
    
    #         # covariance approx like curve_fit
    #         if res.jac is not None and res.jac.size:
    #             try:
    #                 dof = max(1, res.fun.size - res.x.size)
    #                 s_sq = (res.fun @ res.fun) / dof
    #                 JTJ = res.jac.T @ res.jac
    #                 cov = s_sq * inv(JTJ)
    #             except LinAlgError:
    #                 cov = np.full((res.x.size, res.x.size), np.nan)
    #         else:
    #             cov = np.full((len(p0), len(p0)), np.nan)
    
    #         return res.x, cov
    
    #     # --- Original logic with small changes only in the multi-Gauss fit ---
    #     current_peaks = np.array(peaks)
    #     current_peaks = np.append(current_peaks, ind_peak)
    #     current_peaks = np.sort(current_peaks)
    
    #     best_fit_y = None
    #     best_x = None
    #     best_fit_params = None
    #     best_fit_params_error = None
    #     multi_gauss_flag = True
    #     best_idx_interest = None
    #     best_error = np.inf
    
    #     print(current_peaks)
    #     iteration = 0
    #     while len(current_peaks) > 1:
    #         left_boundary, _  = self.calculate_boundaries(x_full, y_full, int(np.min(current_peaks)))
    #         _, right_boundary = self.calculate_boundaries(x_full, y_full, int(np.max(current_peaks)))
    #         x = x_full[left_boundary: right_boundary + 1]
    #         y = y_full[left_boundary: right_boundary + 1]
    
    #         index_of_interest = np.where(current_peaks == ind_peak)[0][0]
    
    #         # Build p0 and bounds (unchanged from your code)
    #         initial_guesses = []
    #         bounds_lower = []
    #         bounds_upper = []
    #         A0_list = []
    #         sig0_list = []
    
    #         for peak in current_peaks:
    #             h, c, w = self.estimate_initial_gaussian_params(x, y, int(peak))
    #             A0 = float(h[0]); mu0 = float(c[0]); sig0 = float(w[0])
    #             initial_guesses.extend([A0, mu0, sig0])
    
    #             # Your original bounds:
    #             lw = 0.1 - sig0 if sig0 > 0.1 else sig0
    #             bounds_lower.extend([0.1 * float(y_full[int(peak)]), float(x_full[int(peak)] - 0.15), lw])
    #             bounds_upper.extend([1.0 + float(y_full[int(peak)]), float(x_full[int(peak)] + 0.15), 0.5 + sig0])
    #             A0_list.append(A0)
    #             sig0_list.append(max(sig0, EPS))
    
    #         bounds = (bounds_lower, bounds_upper)
    
    #         # ---- Multi-Gaussian fit with width/area penalties (simple, no derivatives) ----
    #         try:
    #             lam_sigma = getattr(self, "lambda_sigma", 1e-2)  # small nudge
    #             lam_area  = getattr(self, "lambda_area", 0.0)    # off by default
    #             use_robust = getattr(self, "use_robust", False)
    #             f_scale   = getattr(self, "f_scale", 1.0)
    
    #             popt, pcov = lsq_with_sigma_area_penalty(
    #                 x, y,
    #                 p0=np.array(initial_guesses, float),
    #                 bounds=bounds,
    #                 A0=np.array(A0_list, float),
    #                 sig0=np.array(sig0_list, float),
    #                 lambda_sigma=lam_sigma,
    #                 lambda_area=lam_area,
    #                 robust=use_robust,
    #                 f_scale=f_scale,
    #                 max_nfev=getattr(self, "gi", 1000) * 5)
    #             fitted_y = multigauss(x, popt)
    #             ax.plot(x, fitted_y, alpha=0.4)  # multi-gaussian curve
    
    #             # Keep your RMSE criterion on data only
    #             error = np.sqrt(((fitted_y - y) ** 2).median())
    #             print(error)
    #             print(best_error)
    #             if error < best_error:
    #                 print("improved")
    #                 best_error = error
    #                 best_fit_params = popt
    #                 print(popt)
    #                 print(len(current_peaks))
    #                 best_fit_params_error = pcov
    #                 best_fit_y = fitted_y
    #                 best_x = x
    #                 best_idx_interest = index_of_interest
    #         except Exception as e:
    #             print("Multi-Gaussian penalized fit failed:", e)
    
    #         # prune the farthest neighbor and iterate
    #         distances = np.abs(x[current_peaks] - x_full[ind_peak])
    #         if distances.size > 0:
    #             max_dist_idx = np.argmax(distances)
    #             current_peaks = np.delete(current_peaks, max_dist_idx)
    #         iteration += 1
    
    #     # ---- Single-peak fallback: unchanged ----
    #     if len(current_peaks) == 1:
    #         left_boundary, right_boundary = self.calculate_boundaries(x_full, y_full, ind_peak)
    #         x = x_full[left_boundary: right_boundary + 1]
    #         y = y_full[left_boundary: right_boundary + 1]
    #         height, center, width = self.estimate_initial_gaussian_params(x, y, ind_peak)
    #         height = height[0]; center = center[0]; width = width[0]
    #         initial_decay = 0.1
    #         p0 = [height, center, width, initial_decay]
    #         try:
    #             single_popt, single_pcov = curve_fit(
    #                 lambda x_, amp, cen, wid, dec: self.gaussian_decay(x_, amp, cen, wid, dec),
    #                 x, y, p0=p0, method="trf", maxfev=self.gi)
    #             single_fitted_y = self.gaussian_decay(x, *single_popt)
    #             error = np.sqrt(((single_fitted_y - y) ** 2).median())
    #             print(f"Single error: {error}")
    #             if error < best_error:
    #                 print("Single error better")
    #                 multi_gauss_flag = False
    #                 best_error = error
    #                 best_fit_params = single_popt
    #                 best_fit_params_error = single_pcov
    #                 best_fit_y = single_fitted_y
    #                 best_x = x
    #         except RuntimeError:
    #             print(f"Warning: Optimal parameters could not be found with {self.gi} iterations.")
    
    #     # ---- Finalization: unchanged ----
    #     if multi_gauss_flag is True:
    #         print(f"Selected multi: {trace}")
    #         amp = best_fit_params[best_idx_interest * 3]
    #         cen = best_fit_params[best_idx_interest * 3 + 1]
    #         wid = best_fit_params[best_idx_interest * 3 + 2]
    #         x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, decay=None)
    #         # ax.axvline(x_min, c='red'); ax.axvline(x_max, c='red')
    #         best_x, best_fit_y = self.extrapolate_gaussian(best_x, amp, cen, wid, None, x_min, x_max)
    #         new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
    #         print(best_x)
            
    #         print(best_fit_y)
    #         left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
    #         best_x = best_x[left_boundary - 1: right_boundary + 1]
    #         # ax.axvline(best_x.min(), color="fuchsia"); ax.axvline(best_x.max(), color="fuchsia")
    #         best_fit_y = best_fit_y[left_boundary - 1: right_boundary + 1]
    #         area_smooth, area_ensemble = self.peak_area_distribution(
    #             best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ax, ind_peak, multi=True)
    #     else:
    #         print(f"Selected single: {trace}")
    #         amp, cen, wid, dec = best_fit_params
    #         x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, dec, factor=2)
    #         best_x, best_fit_y = self.extrapolate_gaussian(best_x, amp, cen, wid, dec, x_min, x_max)
    #         new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
    #         print(best_x)
    #         print(best_fit_y)
    #         fig = plt.figure()
    #         plt.plot(best_x, best_fit_y)
    #         plt.show()
    #         left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
    #         best_x = best_x[left_boundary - 1: right_boundary + 1]
    #         best_fit_y = best_fit_y[left_boundary - 1: right_boundary + 1]
    #         area_smooth, area_ensemble = self.peak_area_distribution(
    #             best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ax, ind_peak, multi=False
    #         )
    
    #     return best_x, best_fit_y, area_smooth, area_ensemble
    def _sigma_from_curvature(self, xv, yv, i_local, eps=1e-12):
         """Estimate local sigma from smoothed 2nd derivative at a local apex index i_local."""
         import numpy as np
     
         # >>> Make sure we index positionally, not by pandas labels
         xv = np.asarray(xv)
         yv = np.asarray(yv)
     
         if i_local < 1 or i_local > len(xv) - 2:
             return None
     
         # Smooth y (your smoother can accept arrays)
         ys = self.smoother(yv)
     
         dx1 = xv[i_local]   - xv[i_local - 1]
         dx2 = xv[i_local+1] - xv[i_local]
         ypp = 2.0 * ((ys[i_local + 1] - ys[i_local]) / (dx2 + eps)
                      - (ys[i_local] - ys[i_local - 1]) / (dx1 + eps)) / (dx1 + dx2 + eps)
     
         A = max(ys[i_local], eps)
         if not np.isfinite(ypp) or ypp >= -eps:
             return None
         return float(np.sqrt(A / (-ypp + eps)))
    def _width_seed_and_bounds(self, x_seg, y_seg, c0, cL, cR, w_min_abs, w_max_abs, gap_gamma=0.6):
         """Seed width from curvature and cap by nearest gap in *segment* x-units."""
         import numpy as np
         # >>> Ensure positional math
         x_arr = np.asarray(x_seg)
         y_arr = np.asarray(y_seg)
         
         # positional index of c0 in the segment
         i_loc = int(np.argmin(np.abs(x_arr - c0)))
         
         s_curv = self._sigma_from_curvature(x_arr, y_arr, i_loc)
         
         gapL = abs(c0 - cL) if cL is not None else np.inf
         gapR = abs(c0 - cR) if cR is not None else np.inf
         nearest_gap = np.nanmin([gapL, gapR])
         cap = gap_gamma * nearest_gap if np.isfinite(nearest_gap) else np.inf
         
         hi = np.nanmin([cap, w_max_abs]) if np.isfinite(cap) else w_max_abs
         w0 = hi if (s_curv is None or not np.isfinite(s_curv)) else min(s_curv, hi)
         w0 = float(np.clip(w0, w_min_abs, hi))
         return w0, float(w_min_abs), float(hi)
    
    def fit_gaussians(self, x_full, y_full, ind_peak, trace, peaks, ax, valleys):
       """
       Fits single or multi-Gaussian models to the provided data to determine the best-fit parameters
       for the peak of interest. (Structure preserved; multi-Gaussian seeding/bounds improved.)
       """
    
       # curvature-based sigma + gap cap

    
       # detect overlapping peaks
       current_peaks = np.array(peaks)
       current_peaks = np.append(current_peaks, ind_peak)
       current_peaks = np.sort(current_peaks)
    
       iteration = 0
       best_fit_y = None
       best_x = None
       best_fit_params = None
       best_fit_params_error = None
       best_idx_interest = None
       best_error = np.inf
       multi_gauss_flag = True  # assume multi until single wins
    
       dx_med_full = np.median(np.diff(x_full)) if x_full.size > 1 else 1.0
       if not np.isfinite(dx_med_full) or dx_med_full <= 0:
           dx_med_full = 1.0
    
       # iterate: fit multi-gaussian, drop farthest peak each time
       while len(current_peaks) > 1:
           left_boundary, _ = self.calculate_boundaries(x_full, y_full, np.min(current_peaks))
           _, right_boundary = self.calculate_boundaries(x_full, y_full, np.max(current_peaks))
           x = x_full[left_boundary : right_boundary + 1]
           y = y_full[left_boundary : right_boundary + 1]
    
           # sort by x within the segment
           sort_idx = np.argsort(x_full[current_peaks])
           current_peaks = current_peaks[sort_idx]
    
           # local indices & neighbor coords (for gap-aware caps)
           x_centers_seed = x_full[current_peaks].to_numpy()
           neighbors_L = [x_centers_seed[k-1] if k-1 >= 0 else None for k in range(len(current_peaks))]
           neighbors_R = [x_centers_seed[k+1] if k+1 < len(current_peaks) else None for k in range(len(current_peaks))]
    
           # segment-based width floors/ceilings
           dx_med_seg = np.median(np.diff(x)) if x.size > 1 else dx_med_full
           if not np.isfinite(dx_med_seg) or dx_med_seg <= 0:
               dx_med_seg = dx_med_full
           w_min_abs = max(1.5 * dx_med_seg, 1e-3)                 # floor like your old logic but safer
           w_max_abs = max((x.max() - x.min()) / 3.0, 2 * w_min_abs)
    
           index_of_interest = np.where(current_peaks == ind_peak)[0][0]
    
           initial_guesses = []
           bounds_lower = []
           bounds_upper = []
    
           # Build bounds for fitting
           for k, peak in enumerate(current_peaks):
               # your original seeding (height, center) via helper
               h0, c0, w_est = self.estimate_initial_gaussian_params(x, y, peak)
               h0 = float(h0[0]); c0 = float(c0[0]); w_est = float(w_est[0])
    
               # curvature+gap width (for sandwiched peaks); edges still allowed to be broader
               w0, w_lo, w_hi = self._width_seed_and_bounds(
                   x_seg=x, y_seg=y, c0=c0,
                   cL=neighbors_L[k], cR=neighbors_R[k],
                   w_min_abs=w_min_abs, w_max_abs=w_max_abs,
                   gap_gamma=0.6)
    
               # amplitudes
               a_lo = 0.0
               a_hi = max(1.0 + y_full[peak], h0 * 3.0)
    
               # center bounds
               c_lo = c0 - 0.15
               c_hi = c0 + 0.15
    
               initial_guesses.extend([h0, c0, w0])
               bounds_lower.extend([a_lo, c_lo, w_lo])
               bounds_upper.extend([a_hi, c_hi, w_hi])
    
           bounds = (bounds_lower, bounds_upper)
    
           try:
               popt, pcov = curve_fit(self.multigaussian, x, y, p0=initial_guesses,method="trf", bounds=bounds, maxfev=self.gi)
               fitted_y = self.multigaussian(x, *popt)
               # robustly find which fitted component corresponds to the clicked apex
               fitted_centers = np.array(popt[1::3], float)
               index_of_interest_fit = int(np.argmin(np.abs(fitted_centers - x_full[ind_peak])))
    
               error = float(np.sqrt(((fitted_y - y) ** 2).mean()))  # RMSE
               # print(f"multi error: {error}")
               if error < best_error:
                   
                   best_error = error
                   best_fit_params = popt
                   best_fit_params_error = pcov
                   best_fit_y = fitted_y
                   best_x = x
                   ax.axvline(x_full[index_of_interest_fit])
                   best_idx_interest = index_of_interest_fit
           except RuntimeError:
               pass
    
           # remove the farthest peak from the clicked apex (preserve your strategy)
           distances = np.abs(x[current_peaks] - x_full[ind_peak])
           if distances.size > 0:
               max_dist_idx = int(np.argmax(distances))
               current_peaks = np.delete(current_peaks, max_dist_idx)
           iteration += 1
    
       # Single gaussian fit with decay
       # if len(current_peaks) == 1:
       #     left_boundary, right_boundary = self.calculate_boundaries(x_full, y_full, ind_peak)
       #     x = x_full[left_boundary : right_boundary + 1]
       #     y = y_full[left_boundary : right_boundary + 1]
       #     height, center, width = self.estimate_initial_gaussian_params(x, y, ind_peak)
       #     height = float(height[0]); center = float(center[0]); width = float(width[0])
       #     # initial_decay = 0.1
       #     # p0 = [height, center, width, initial_decay]
       #     p0 = [height, center, width]
       #     try:
       #         single_popt, single_pcov = curve_fit(
       #             # lambda xv, amp, cen, wid, dec: self.gaussian_decay(xv, amp, cen, wid, dec),
       #             # x, y, p0=p0, method="trf", maxfev=self.gi)
       #             lambda xv, amp, cen, wid: self.individual_gaussian(xv, amp, cen, wid),
       #             x, y, p0=p0, method="trf", maxfev=self.gi)
       #         # single_fitted_y = self.gaussian_decay(x, *single_popt)
       #         single_fitted_y = self.individual_gaussian(x, *single_popt)
       #         error = float(np.sqrt(((single_fitted_y - y) ** 2).mean()))  # RMSE
       #         # print(f"single error: {error}")
       #         if error < best_error/1.02:
       #             multi_gauss_flag = False
       #             best_error = error
       #             best_fit_params = single_popt
       #             best_fit_params_error = single_pcov
       #             best_fit_y = single_fitted_y
       #             best_x = x
       #             best_idx_interest = 0  # single component
       #     except RuntimeError:
       #         pass
# ------------------ Single gaussian fit (use full-trace peaks for valleys; choose closer of boundary vs valley) -----------------
       if len(current_peaks) == 1:
            apex_x   = float(x_full[ind_peak])
            full_peaks = np.asarray(self.peaks[trace])
        
            # 1) Start with your original derivative-based boundaries
            left_b1, right_b1 = self.calculate_boundaries(x_full, y_full, ind_peak)
            L_idx, R_idx = int(left_b1), int(right_b1)

            v2v = self.find_valleys(y_full, full_peaks, peak_oi = ind_peak)
            vL_idx = None
            vR_idx = None
            
            if isinstance(v2v, (list, tuple)) and len(v2v) == 2:
                vL_candidate, vR_candidate = int(v2v[0]), int(v2v[1])
            
                # keep only if they exist and are on the correct sides of the apex
                if 0 <= vL_candidate < ind_peak:
                    vL_idx = vL_candidate
                if ind_peak < vR_candidate < len(x_full):
                    vR_idx = vR_candidate
        
            def _closer(idx_a, idx_b, apex_x_val):
                """Return whichever index is closer in x to apex; if one is None, return the other."""
                if idx_a is None and idx_b is None:
                    return None
                if idx_a is None:
                    return idx_b
                if idx_b is None:
                    return idx_a
                xa = float(x_full[idx_a]); xb = float(x_full[idx_b])
                return idx_a if abs(xa - apex_x_val) <= abs(xb - apex_x_val) else idx_b
            
            # left side (must be strictly < apex)
            cand_left = _closer(L_idx, vL_idx, apex_x)
            if cand_left is not None and cand_left < ind_peak:
                L_idx = int(cand_left)
            
            # right side (must be strictly > apex)
            cand_right = _closer(R_idx, vR_idx, apex_x)
            if cand_right is not None and cand_right > ind_peak:
                R_idx = int(cand_right)
            
            # final sanity: valid span; otherwise fall back to original derivative bounds
            if not (0 <= L_idx < R_idx < len(x_full)):
                L_idx, R_idx = int(left_b1), int(right_b1)
            
            # ---------------- window chosen; proceed to fit ----------------
            x = x_full[L_idx : R_idx + 1]
            y = y_full[L_idx : R_idx + 1]
            
            x_vals = np.asarray(x)  # works for Series or ndarray
            i_loc = int(np.argmin(np.abs(x_vals - apex_x)))
            peak_key = x.index[i_loc] if hasattr(x, "index") else i_loc
            # estimator needs Series (uses .iloc), so pass x,y as-is here
            height, center, width = self.estimate_initial_gaussian_params(x, y, peak_key)
            amp0 = float(height[0]); wid0 = float(width[0])
            
            # NOW convert to NumPy for curve_fit and downstream numeric ops
            if hasattr(x, "to_numpy"): x = x.to_numpy()
            if hasattr(y, "to_numpy"): y = y.to_numpy()
            
            # 6) Width bounds (rest of your code unchanged below)
            dx = np.diff(x)
            dx_med = np.median(dx[np.isfinite(dx)]) if dx.size else 1.0
            w_min_abs = max(1.5 * dx_med, 1e-3)
            w_max_abs = max((x.max() - x.min()) / 3.0, 2 * w_min_abs)
            wid0 = float(np.clip(wid0, w_min_abs, w_max_abs))
            a_hi = max(1.0 + float(y_full[ind_peak]), amp0 * 3.0)
            c_pad = 1e-2
            p0 = [amp0, apex_x, wid0]
            lb = [0.0,        apex_x - c_pad, w_min_abs]
            ub = [a_hi,       apex_x + c_pad, w_max_abs]
        
            try:
                single_popt, single_pcov = curve_fit(
                    lambda xv, a, c, w: self.individual_gaussian(xv, a, c, w),
                    x, y, p0=p0, bounds=(lb, ub), method="trf", maxfev=self.gi
                )
                single_fitted_y = self.individual_gaussian(x, *single_popt)
                error = float(np.sqrt(((single_fitted_y - y) ** 2).mean()))
        
                # keep single only if it beats multi by your margin
                if error < best_error / 1.02:
                    multi_gauss_flag = False
                    best_error = error
                    best_fit_params = single_popt
                    best_fit_params_error = single_pcov
                    best_fit_y = single_fitted_y
                    best_x = x
                    best_idx_interest = 0
            except RuntimeError:
                pass
       # post-processing: extend -> boundaries -> slice -> area
       if multi_gauss_flag is True:
           print(f"Selected multi: {trace}")
           amp = best_fit_params[best_idx_interest * 3 + 0]
           cen = best_fit_params[best_idx_interest * 3 + 1]
           wid = best_fit_params[best_idx_interest * 3 + 2]
    
           # principled extension (decay=None for symmetric)
           x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, decay=None)
           best_x, best_fit_y = self.extrapolate_gaussian(best_x, amp, cen, wid, None, x_min, x_max)
           new_ind_peak = int(np.abs(best_x - x_full[ind_peak]).argmin())
           left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
           best_x = best_x[left_boundary - 1 : right_boundary + 1]
           best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
           # area_smooth, area_ensemble = self.peak_area_distribution(
           #     best_fit_params, best_fit_params_error, best_idx_interest,
           #     best_x, x_full, ax, ind_peak, multi=True)
           area_smooth, area_ensemble = self.peak_area_distribution(
               best_fit_params, best_fit_params_error, best_idx_interest,
               best_x, x_full, ax, ind_peak, best_fit_y, step = 0.001, min_pts = 21)
    
       else:
           print(f"Selected single: {trace}")
           # amp, cen, wid, dec = best_fit_params[0], best_fit_params[1], best_fit_params[2], best_fit_params[3]
           amp, cen, wid  = best_fit_params[0], best_fit_params[1], best_fit_params[2]
           # x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, dec, factor=2)
           x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, decay=None, factor=2)
           # best_x, best_fit_y = self.extrapolate_gaussian(best_x, amp, cen, wid, dec, x_min, x_max)
           best_x, best_fit_y = self.extrapolate_gaussian(best_x, amp, cen, wid, None, x_min, x_max)
           new_ind_peak = int(np.abs(best_x - x_full[ind_peak]).argmin())
           left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
           best_x = best_x[left_boundary - 1 : right_boundary + 1]
           best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
           # area_smooth, area_ensemble = self.peak_area_distribution(
           #     best_fit_params, best_fit_params_error, best_idx_interest,
           #     best_x, x_full, ax, ind_peak, multi=False)
           area_smooth, area_ensemble = self.peak_area_distribution(
               best_fit_params, best_fit_params_error, best_idx_interest,
               best_x, x_full, ax, ind_peak, best_fit_y, step = 0.001, min_pts = 21)
       return best_x, best_fit_y, area_smooth, area_ensemble
    

    # def peak_area_distribution(self, params, params_uncertainty, ind, x, x_full, ax, ind_peak, multi, n_samples= 1000):
    #     area_ensemble = []

    #     if multi:
    #         amp_i, cen_i, wid_i = params[ind*3], params[ind*3+1], params[ind*3+2]
    #         start, end = 3*ind, 3*ind+3
    #         pcov = params_uncertainty[start:end, start:end]
        
    #         # Sample with wid in log space
    #         samples = self._sample_mvnorm_with_log(
    #             mean=np.array([amp_i, cen_i, wid_i]),
    #             cov=pcov,
    #             log_idx=[2],                # wid only
    #             n=n_samples)
        
    #         for i in range(n_samples):
    #             amp, cen, wid = samples[i]
    #             # (Optional) enforce non-negative amplitude
    #             if amp < 0: 
    #                 amp = 0.0
        
    #             best_fit_y = self.individual_gaussian(x, amp, cen, wid)
    #             best_x, best_fit_y = self.extrapolate_gaussian(
    #                 x, amp, cen, wid, None, x.min()-1, x.max()+1, step=0.01)
    #             if best_x.size == 0: 
    #                 continue  # skip pathological draw
        
    #             new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
    #             left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
    #             L = max(left_boundary - 1, 0)
    #             R = min(right_boundary + 1, best_x.size - 1)
    #             if R < L: 
    #                 continue
    #             best_x = best_x[L:R+1]
    #             best_fit_y = np.maximum(best_fit_y[L:R+1], 0)
    #             area_ensemble.append(simpson(y=best_fit_y, x=best_x))
    #         if not area_ensemble:
    #             raise ValueError("area_ensemble ended up empty, cannot compute mean")
    #         return np.mean(area_ensemble), area_ensemble
    #     else:
    #         # params is [amp, cen, wid, dec]; params_uncertainty is 4x4
    #         samples = self._sample_mvnorm_with_log(
    #             mean=np.asarray(params, float),
    #             cov=np.asarray(params_uncertainty, float),
    #             log_idx=[2, 3],            # wid and decay in log space
    #             n=n_samples)
        
    #         for i in range(n_samples):
    #             amp, cen, wid, dec = samples[i]
    #             if amp < 0:
    #                 amp = 0.0  # optional clamp
        
    #             x_min, x_max = self.calculate_gaus_extension_limits(cen, wid, dec, factor=2)
    #             # quick sanity on bounds
    #             if not (np.isfinite(x_min) and np.isfinite(x_max)) or x_max <= x_min:
    #                 continue
        
    #             best_x, best_fit_y = self.extrapolate_gaussian(
    #                 x, amp, cen, wid, dec, x_min, x_max, step=0.01)
    #             if best_x.size == 0:
    #                 continue
        
    #             new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
    #             left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
    #             L = max(left_boundary - 1, 0)
    #             R = min(right_boundary + 1, best_x.size - 1)
    #             if R < L:
    #                 continue
    #             best_x = best_x[L:R+1]
    #             best_fit_y = np.maximum(best_fit_y[L:R+1], 0)
    #             area_ensemble.append(simpson(y=best_fit_y, x=best_x))
    #         if not area_ensemble:
    #             raise ValueError("area_ensemble ended up empty, cannot compute mean")
    #         return np.mean(area_ensemble), area_ensemble

    def peak_area_distribution(
        self,
        best_fit_params,
        best_fit_params_error,
        best_idx_interest,
        best_x,
        x_full,      # kept for API compatibility (not used here)
        ax,                      # kept for API compatibility (optional plotting)
        ind_peak,           # kept for API compatibility (not used here)
        best_fit_y,
        *,
        step,
        min_pts):
        """
        Compute area for the *already-selected & already-sliced* peak window.
        No boundary finding here. No 'multi' branching. Just integrate.
    
        Parameters
        ----------
        best_fit_params : array
            If len % 3 == 0 → multi-Gaussian params [a1,c1,w1, a2,c2,w2, ...]
            Else (len == 4) → single with decay [amp, cen, wid, dec].
        best_fit_params_error : array or None
            Curve-fit covariance. If present, we’ll do a light uncertainty ensemble.
        best_idx_interest : int
            Index of the selected component (for multi); ignored for single.
        best_x : array
            X grid for the selected component (already extended & cropped).
        x_full : array
            Unused (kept to avoid touching all call sites).
        ax : matplotlib axis or None
            Optional plotting (no effect if None).
        ind_peak : int
            Unused (kept to avoid touching all call sites).
        best_fit_y : array or None
            If None, we reconstruct the selected component’s Y on `best_x`.
            If provided, we trust it.
        step : float
            Only used if we must reconstruct and `best_x` is tiny.
        min_pts : int
            Minimum number of samples to integrate robustly.
    
        Returns
        -------
        area_smooth : float
            Simpson integration of the selected component over `best_x`.
        area_ensemble : float
            If covariance available: mean area from a small param ensemble.
            Else: equals `area_smooth`.
        """
        best_x = np.asarray(best_x, float)
        best_fit_y = np.asarray(best_fit_y, float)
    
        # 1) Guard against degenerate windows: ensure at least `min_pts`
        if best_x.size < min_pts:
            # Expand symmetrically around the midpoint of current best_x
            x_mid = float(best_x[np.clip(best_x.size // 2, 0, best_x.size - 1)])
            pad = (min_pts - max(best_x.size, 2)) * step / 2.0
            x_min = x_mid - pad
            x_max = x_mid + pad
    
            # Re-generate best_x and best_fit_y on the expanded grid using the same params
            new_x = np.arange(x_min, x_max + 0.5 * step, step)
            if len(best_fit_params) % 3 == 0:
                a = best_fit_params[3 * best_idx_interest + 0]
                c = best_fit_params[3 * best_idx_interest + 1]
                w = best_fit_params[3 * best_idx_interest + 2]
                new_y = self.individual_gaussian(new_x, a, c, w)
            else:
                a, c, w, d = best_fit_params
                new_y = self.gaussian_decay(new_x, a, c, w, d)
    
            best_x, best_fit_y = new_x, new_y
    
        # 2) Ensure non-negativity & finite values
        y = np.nan_to_num(best_fit_y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.maximum(y, 0.0)
    
        # 3) Integrate
        area_smooth = float(simpson(y, best_x)) if best_x.size >= 2 else 0.0
    
        # 4) Light ensemble (optional): if covariance available, propagate into area
        #    We restrict to the selected component’s block for multi.
        area_ensemble = area_smooth
        try:
            pcov = np.asarray(best_fit_params_error)
            if pcov.ndim == 2 and pcov.shape[0] == pcov.shape[1]:
                if len(best_fit_params) % 3 == 0:
                    # sub-covariance for [a,c,w] of the selected component
                    i0 = 3 * best_idx_interest
                    sub = pcov[i0:i0 + 3, i0:i0 + 3]
                    mu = np.array([a, c, w], float)
                    draws = np.random.multivariate_normal(mu, sub, size=128)
                    areas = []
                    for A, C, W in draws:
                        W = max(W, 1e-9)  # keep positive
                        yi = self.individual_gaussian(best_x, max(A, 0.0), C, W)
                        yi = np.maximum(np.nan_to_num(yi, nan=0.0), 0.0)
                        areas.append(simpson(yi, best_x))
                    if areas:
                        area_ensemble = float(np.mean(areas))
                else:
                    # single w/decay: if 4x4 cov is present, sample [a,c,w,d]
                    if pcov.shape == (4, 4):
                        mu = np.array([a, c, w, d], float)
                        draws = np.random.multivariate_normal(mu, pcov, size=128)
                        areas = []
                        for A, C, W, D in draws:
                            W = max(W, 1e-9)
                            yi = self.gaussian_decay(best_x, max(A, 0.0), C, W, max(D, 0.0))
                            yi = np.maximum(np.nan_to_num(yi, nan=0.0), 0.0)
                            areas.append(simpson(yi, best_x))
                        if areas:
                            area_ensemble = float(np.mean(areas))
        except Exception:
            # If anything goes sideways with covariance sampling, just fall back gracefully
            area_ensemble = area_smooth
    
        return area_smooth, area_ensemble
    # def _mvnorm_to_logspace(self, mean, cov, log_idx):
    #     """
    #     First-order transform (delta method) to log-parameter space.
    #     mean: 1D array of parameters in original space
    #     cov : covariance in original space
    #     log_idx: indices to be log-transformed (e.g., [2] or [2,3])
    #     Returns (mean_log, cov_log).
    #     """
    #     mean = np.asarray(mean, float).copy()
    #     cov = np.asarray(cov, float).copy()
    #     eps = 1e-12
    #     # Ensure positivity before taking logs
    #     for j in log_idx:
    #         if not np.isfinite(mean[j]) or mean[j] <= eps:
    #             mean[j] = max(eps, abs(mean[j]))
    #     mean_log = mean.copy()
    #     for j in log_idx:
    #         mean_log[j] = np.log(mean[j])
    #     # Jacobian of (id for linear dims, d/dx log x = 1/x for log dims)
    #     J = np.eye(mean.size)
    #     for j in log_idx:
    #         J[j, j] = 1.0 / mean[j]
    #     cov_log = J @ cov @ J.T
    #     return mean_log, cov_log
    def _mvnorm_to_logspace(self, mean, cov, log_idx):
        """
        First-order transform (delta method) to log-parameter space.
        mean: 1D array of parameters in original space
        cov : covariance in original space
        log_idx: indices to be log-transformed (e.g., [2] or [2,3])
        Returns (mean_log, cov_log) stabilized.
        """
        mean = np.asarray(mean, float).copy()
        cov  = np.asarray(cov,  float).copy()
    
        # Clean covariance (symmetrize, NaNs->0)
        cov = np.nan_to_num(0.5 * (cov + cov.T), nan=0.0, posinf=0.0, neginf=0.0)
    
        # Ensure strictly positive means for log dims; avoid huge 1/mean
        eps_pos = 1e-6
        min_for_jac = 1e-3   # cap 1/mean at <= 1/1e-3 = 1e3
        for j in log_idx:
            if not np.isfinite(mean[j]) or mean[j] <= eps_pos:
                # If mean is tiny or non-finite, push to small positive
                mean[j] = eps_pos
    
        # Build Jacobian
        J = np.eye(mean.size, dtype=float)
        for j in log_idx:
            denom = max(min_for_jac, mean[j])   # prevents exploding 1/mean
            J[j, j] = 1.0 / denom
    
        # Map to log-covariance
        cov_log = J @ cov @ J.T
        cov_log = np.nan_to_num(0.5 * (cov_log + cov_log.T), nan=0.0, posinf=0.0, neginf=0.0)
    
        # Make PSD (clip small negative eigs)
        w, V = np.linalg.eigh(cov_log)
        w_clipped = np.clip(w, 1e-12, None)
        cov_log_psd = (V * w_clipped) @ V.T
    
        # Mean in log space
        mean_log = mean.copy()
        for j in log_idx:
            mean_log[j] = np.log(max(mean[j], eps_pos))
    
        return mean_log, cov_log_psd
    
    
    # def _sample_mvnorm_with_log(self, mean, cov, log_idx, n):
    #     """
    #     Draw samples in log space for indices in log_idx, then map back.
    #     """
    #     m_log, C_log = self._mvnorm_to_logspace(mean, cov, log_idx)
    #     # Draw in transformed space
    #     s_log = np.random.multivariate_normal(
    #         m_log, C_log, size=n, check_valid='ignore', tol=1e-8)
    #     # Map back
    #     s = s_log.copy()
    #     for j in log_idx:
    #         s[:, j] = np.exp(s_log[:, j])
    #     return s
    
    def _sample_mvnorm_with_log(self, mean, cov, log_idx, n, x_domain=None):
        """
        Draw samples in log space for indices in log_idx, then map back with bounds.
        x_domain: tuple (x_min, x_max) to derive sensible width bounds.
        """
        m_log, C_log = self._mvnorm_to_logspace(mean, cov, log_idx)
    
        # --- Physical bounds (in ORIGINAL space) ---
        # Width (sigma) bounds from x-domain:
        if x_domain is not None:
            x_min, x_max = float(x_domain[0]), float(x_domain[1])
            span = max(1e-6, x_max - x_min)
            # resolution proxy
            dx_med = max(1e-6, float(np.median(np.diff(np.linspace(x_min, x_max, 100)))))
            wid_min = max(3 * dx_med, 1e-4 * span)     # not below 3 samples or 0.01% of span
            wid_max = 0.5 * span                       # not wider than half the window
        else:
            wid_min, wid_max = 1e-4, 1.0
    
        # Decay (positive, avoid near-0 -> huge 1/decay tails)
        dec_min, dec_max = 1e-3, 5.0
    
        # Build per-dimension bounds in ORIGINAL space
        bounds_lo = np.full(mean.shape, -np.inf)
        bounds_hi = np.full(mean.shape,  np.inf)
        for j in log_idx:
            # if this index corresponds to width (by your API, wid is index 2)
            if j == 2:
                bounds_lo[j], bounds_hi[j] = wid_min, wid_max
            # if this index corresponds to decay (index 3 in your single-peak case)
            if j == 3:
                bounds_lo[j], bounds_hi[j] = dec_min, dec_max
    
        # Convert bounds to LOG space for clipping
        log_lo = np.where(np.isfinite(bounds_lo), np.log(np.maximum(bounds_lo, 1e-12)), -np.inf)
        log_hi = np.where(np.isfinite(bounds_hi), np.log(np.maximum(bounds_hi, 1e-12)),  np.inf)
    
        # --- Draws in log space, with robust fallbacks ---
        try:
            s_log = np.random.multivariate_normal(m_log, C_log, size=n, check_valid='ignore', tol=1e-8)
        except Exception:
            # Fallback: diagonalized, shrunken covariance
            diag = np.clip(np.diag(C_log), 1e-12, None)
            C_fallback = np.diag(diag * 0.25)  # shrink variance
            s_log = np.random.multivariate_normal(m_log, C_fallback, size=n, check_valid='ignore', tol=1e-8)
    
        # Clip to bounds in log space to prevent overflow / unphysical values
        for j in log_idx:
            lo = log_lo[j] if np.isfinite(log_lo[j]) else -np.inf
            hi = log_hi[j] if np.isfinite(log_hi[j]) else  np.inf
            s_log[:, j] = np.clip(s_log[:, j], lo, hi)
    
        # Map back
        s = s_log.copy()
        for j in log_idx:
            s[:, j] = np.exp(s_log[:, j])
    
        return s
    
    def handle_peak_selection(self, ax, ax_idx, xdata, y_bcorr, peak_idx, peaks, trace):
        """
        Handles the selection of a peak, fits a Gaussian to the selected peak, and updates the plot and internal data structures.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis object on which the peak and Gaussian fit will be plotted.
        ax_idx : int
            Index of the subplot or axis where the peak is selected.
        xdata : numpy.ndarray or pandas.Series
            Array of x-values (e.g., retention times) corresponding to the data points.
        y_bcorr : numpy.ndarray or pandas.Series
            Array of baseline-corrected y-values (e.g., signal intensities) corresponding to the x-values.
        peak_idx : int
            Index of the selected peak in the data.
        peaks : list of int
            List of indices representing the detected peaks in the data.
        trace : str
            Identifier for the trace being analyzed (e.g., which sample or dataset the peak belongs to).

        Returns
        -------
        None
            This function updates the plot and internal data structures but does not return any values.

        Notes
        -----
        - The function identifies valleys around the selected peak and determines the neighborhood boundaries for peak fitting.
        - A Gaussian fit is applied to the selected peak and its neighborhood.
        - The area under the Gaussian curve is calculated and displayed on the plot along with retention time.
        - The peak integration results are stored in `self.integrated_peaks` and `self.peak_results` for later analysis.
        - If the peak selection or fitting process encounters a runtime error, the exception is handled and ignored.
        """
        try:
            valleys = self.find_valleys(y_bcorr, peaks)
            A, B, peak_neighborhood = self.find_peak_neighborhood_boundaries(xdata, y_bcorr, self.peaks[trace], valleys, peak_idx, ax, self.max_peaks_for_neighborhood, trace)
            x_fit, y_fit_smooth, area_smooth, area_ensemble = self.fit_gaussians(xdata, y_bcorr, peak_idx, trace, peak_neighborhood, ax, valleys)
            fill = ax.fill_between(x_fit, 0, y_fit_smooth, color="grey", alpha=0.5)
            rt_of_peak = xdata[peak_idx]
            area_text = f"Area: {area_smooth:.0f}\nRT: {rt_of_peak:.0f}"
            text_annotation = ax.annotate(area_text, xy=(rt_of_peak + 1.5, y_fit_smooth.max() * 0.5), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="grey")
            self.integrated_peaks[(ax_idx, peak_idx)] = {"fill": fill, "area": area_smooth, "rt": rt_of_peak, "text": text_annotation, "trace": trace, 'area_ensemble': area_ensemble}
            plt.draw()
            if trace not in self.peak_results:
                self.peak_results[trace] = {"rts": [], "areas": [], "area_ensemble": []}
            self.peak_results[trace]["rts"].append(rt_of_peak)
            self.peak_results[trace]["areas"].append(area_smooth)  # Calculate area if needed
            self.peak_results[trace]["area_ensemble"].append(area_ensemble)
        except RuntimeError:
            pass

    ######################################################
    ################      Plot      ######################
    ######################################################
    def add_window_controls(self):
        """
        Adds interactive TextBox widgets to the existing figure so that the user can change the x-window.
        This method does not re-create the entire plot.
        """
        # If controls already exist, do nothing or toggle visibility.
        if hasattr(self, "window_controls_added") and self.window_controls_added:
            return

        fig = self.fig  # Use the stored figure

        # Create new axes for the text boxes in normalized coordinates
        axbox_min = fig.add_axes([0.25, 0.025, 0.05, 0.02])
        axbox_max = fig.add_axes([0.65, 0.025, 0.05, 0.02])
        text_box_min = TextBox(axbox_min, 'Window start: ', initial=str(self.window_bounds[0]))
        text_box_max = TextBox(axbox_max, 'Window end: ', initial=str(self.window_bounds[1]))

        def submit_callback(text):
            try:
                new_xmin = float(text_box_min.text)
                new_xmax = float(text_box_max.text)
                self.window_bounds = [new_xmin, new_xmax]
                # For each subplot, update the x-limits (and update y-limits if desired)
                for i, ax in enumerate(self.axs):
                    # If you are updating data based on window bounds, you can filter your full data here.
                    # Otherwise, simply update the limits.
                    ax.set_xlim(self.window_bounds)
                    # Optionally, update y-limits based on filtered data.
                fig.canvas.draw_idle()
            except Exception as e:
                print("Invalid input for window boundaries:", e)

        text_box_min.on_submit(submit_callback)
        text_box_max.on_submit(submit_callback)
        self.window_controls_added = True
        plt.draw()

    def plot_data(self):
        """
        Creates subplots for each trace and adds two text boxes to allow the user to update the x-window boundaries.
        """
        # Create subplots as before
        if len(self.traces) == 1:
            fig, ax = plt.subplots(figsize=(8, 10))
            axs = [ax]
        else:
            fig, axs = plt.subplots(len(self.traces), 1, figsize=(8, 10), sharex=True)
            axs = axs.ravel()

        # Initialize storage for datasets and peak indices if not already
        self.datasets = [None] * len(self.traces)
        self.peaks_indices = [None] * len(self.traces)

        # Create the subplots and store the full data and line objects
        for i, ax in enumerate(axs):
            self.setup_subplot(ax, i)
            if i == len(self.traces) - 1:
                ax.set_xlabel("Corrected Retention Time (minutes)")

        fig.suptitle(f"Sample: {self.sample_name}", fontsize=16, fontweight="bold")

        return fig, axs

    def setup_subplot(self, ax, trace_idx):
        """
        Configures a single subplot for the given trace, computes and stores the full
        processed data, then plots it.
        """
        # Get full x-values and y-data for the trace
        x_values = self.df["rt_corr"]
        trace = self.traces[trace_idx]
        y = self.df[trace]

        # Baseline correction and smoothing on the full dataset
        y_base, min_peak_amp = self.baseline(x_values, y)
        y_bcorr = y - y_base
        y_bcorr[y_bcorr < 0] = 0
        y_filtered = self.smoother(y_bcorr)

        # Store the full processed data for later updates
        if not hasattr(self, "full_data"):
            self.full_data = {}
        self.full_data[trace_idx] = (x_values, y_filtered)

        # Plot the full data; even if the current x-limits are restricted, we plot everything
        ax.plot(x_values, y, c='grey', alpha=0.3)
        line, = ax.plot(x_values, y_filtered, "k")
        if not hasattr(self, "line_objects"):
            self.line_objects = {}
        self.line_objects[trace_idx] = line

        # Set the current x-limits based on the current window_bounds
        ax.set_xlim(self.window_bounds)
        ax.set_ylabel(trace)

        # Adjust y-limits based on data within the current window
        within_xlim = (x_values >= self.window_bounds[0]) & (x_values <= self.window_bounds[1])
        y_within = y_filtered[within_xlim]
        if len(y_within) > 0:
            ymin, ymax = y_within.min(), y_within.max()
            y_margin = (ymax - ymin) * 0.1  # 10% margin
            ax.set_ylim(0, ymax + y_margin)
        else:
            ax.set_ylim(0, 1)

        # Store additional info for peak selection, etc.
        self.axs_to_traces[ax] = trace
        self.datasets[trace_idx] = (x_values, y_bcorr)
        if self.max_peak_amp is not None:
            peaks_total, properties = find_peaks(y_filtered, height=(min_peak_amp, self.max_peak_amp), prominence=self.pk_pr)
        else:
            peaks_total, properties = find_peaks(y_filtered, height=min_peak_amp, prominence=self.pk_pr)
        self.peaks[trace] = peaks_total
        self.peak_properties[trace] = properties
        self.peaks_indices[trace_idx] = peaks_total
# Baseline
    # def _robust_sigma_amp(self, c):
    #     # robust noise from first difference
    #     d = np.diff(np.asarray(c, float))
    #     mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    #     sigma_amp = mad / np.sqrt(2.0)
    #     if not np.isfinite(sigma_amp) or sigma_amp <= 0:
    #         sigma_amp = 0.0
    #     return float(sigma_amp)
    
    # def _estimate_typical_width_points(self, x, y):
    #     """
    #     Two-pass: conservative SNIP-A -> corrected -> widths from find_peaks.
    #     Returns median width in SAMPLES, or None if not enough peaks.
    #     """
    #     # SNIP-A: conservative, length-based
    #     bA, cA = self.snip_baseline(y, peak_width_points=None, max_iter=None, pad_edges=True)
    
    #     # Smooth a bit for peak finding stability (you already have smoother)
    #     cS = self.smoother(np.clip(cA, 0, None))
    
    #     # Prominence from noise on corrected signal
    #     sig = self._robust_sigma_amp(cS)
    #     prom = max(5.0 * sig, 1e-12)  # k=5 default
    
    #     # Find peaks and widths at 50% height
    #     peaks, props = find_peaks(cS, prominence=prom)
    #     if peaks.size < 3:
    #         # Try a touch less strict if nothing shows
    #         peaks, props = find_peaks(cS, prominence=max(3.0 * sig, 1e-12))
    #     if peaks.size < 3:
    #         return None
    
    #     w, _, _, _ = peak_widths(cS, peaks, rel_height=0.5)  # width in samples
    #     w = w[np.isfinite(w) & (w > 1)]
    #     if w.size == 0:
    #         return None
    
    #     # Robust typical width = median with clipping
    #     n = len(y)
    #     w_med = np.median(w)
    #     w_med = int(np.clip(w_med, 10, max(20, n // 10)))  # guardrails
    #     return w_med
        
    # def _lls(self, x):
    #     # Log-Log-Sqrt compression; assumes x >= 0 (clip small negatives to 0)
    #     x = np.clip(x, 0, None)
    #     return np.log(np.log(np.sqrt(x + 1.0) + 1.0) + 1.0)

    # def _inv_lls(self, y):
    #     # Inverse of Log-Log-Sqrt
    #     return (np.exp(np.exp(y) - 1.0) - 1.0)**2 - 1.0

    # def snip_baseline(self, y, *, peak_width_points=None, max_iter=None, pad_edges=True):
    #     y = np.asarray(y, dtype=float)
    
    #     # Fill NaNs
    #     if np.isnan(y).any():
    #         yy = y.copy()
    #         mask = np.isnan(yy)
    #         idx = np.where(~mask, np.arange(len(yy)), 0)
    #         np.maximum.accumulate(idx, out=idx)
    #         yy[mask] = yy[idx[mask]]
    #         if np.isnan(yy[0]):
    #             first = np.flatnonzero(~np.isnan(y))[0]
    #             yy[:first] = yy[first]
    #         y = yy
    
    #     # Compress
    #     z  = self._lls(y)
    #     n  = z.size
    
    #     # Iterations (priority: explicit max_iter > width-based > conservative fallback)
    #     if max_iter is None:
    #         if peak_width_points is not None:
    #             req = int(max(10, round(0.6 * peak_width_points)))   # α=0.6
    #         else:
    #             req = max(20, int(0.002 * n))                        # conservative SNIP-A fallback
    #         max_iter = min(req, max(1, n // 2 - 1))
    
    #     # Optional reflect padding to stabilize edges
    #     if pad_edges and max_iter > 0:
    #         zf = np.pad(z, (max_iter, max_iter), mode="reflect").copy()
    #         Np = zf.size
    #         for m in range(1, max_iter + 1):
    #             interior = Np - 2*m
    #             if interior <= 0: break
    #             left   = zf[:interior]
    #             right  = zf[2*m:2*m+interior]
    #             center = zf[m:m+interior]
    #             zf[m:m+interior] = np.minimum(center, 0.5*(left + right))
    #         zf = zf[max_iter:-max_iter]  # unpad
    #     else:
    #         zf = z.copy()
    #         for m in range(1, max_iter + 1):
    #             interior = n - 2*m
    #             if interior <= 0: break
    #             left   = zf[:interior]
    #             right  = zf[2*m:2*m+interior]
    #             center = zf[m:m+interior]
    #             zf[m:m+interior] = np.minimum(center, 0.5*(left + right))
    
    #     baseline  = self._inv_lls(zf)
    #     baseline  = np.clip(baseline, 0, np.max(y))
    #     corrected = y - baseline
    #     return baseline, corrected

    
    # def baseline(self, x, y, *args, **kwargs):
    #     y = np.asarray(y, float)
    
    #     # choose lam and p from data
    #     dx = np.median(np.diff(np.asarray(x, float))) if len(x) > 1 else 1.0
    #     span = (x.max() - x.min()) if len(x) else 1.0
    
    #     # Rule of thumb:
    #     # - lam controls smoothness (bigger = smoother). Start 1e6–1e7 for LC.
    #     # - p << 0.5 protects positive peaks; 0.001–0.01 typical.
    #     # If sampling is very fine (small dx), increase lam.
    #     lam = 1e6 * max(1.0, (span / max(dx, 1e-6)) / 200.0)   # scale with samples per span
    #     p   = 0.002                                            # aggressive protection of positives
    
    #     b, info = asls(y, lam=lam, p=p, max_iter=50)           # smooth, valley-preserving
    #     b = np.maximum(b, 0.0)
    #     c = np.clip(y - b, 0, None)
    
    #     # robust noise on c for a defensible detection floor
    #     d = np.diff(c)
    #     mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    #     sigma = (mad / np.sqrt(2.0)) if (mad > 0 and np.isfinite(mad)) else 0.0
    #     k = 5.0
    #     dyn = np.nanpercentile(y, 99) - np.nanpercentile(y, 1)
    #     abs_floor = 0.005 * dyn
    #     rel_floor = 0.02 * np.nanmedian(b) if np.isfinite(np.nanmedian(b)) else 0.0
    #     min_peak_amp = max(k * sigma, abs_floor, rel_floor)
    #     return b, float(min_peak_amp*3)
    
    # def baseline(self, x, y, deg=5, max_it=1000, tol=1e-4):
    #     """
    #     Performs baseline correction on the input signal using an iterative polynomial fitting approach.

    #     Parameters
    #     ----------
    #     y : numpy.ndarray or pandas.Series
    #         The input signal (e.g., chromatographic data) that requires baseline correction.
    #     deg : int, optional
    #         The degree of the polynomial used for fitting the baseline. Default is 5.
    #     max_it : int, optional
    #         The maximum number of iterations for the baseline fitting process. Default is 50.
    #     tol : float, optional
    #         The tolerance for stopping the iteration when the change in coefficients becomes small. Default is 1e-4.

    #     Returns
    #     -------
    #     base : numpy.ndarray
    #         The estimated baseline for the input signal.

    #     Notes
    #     -----
    #     - The function iteratively fits a polynomial baseline to the input signal, adjusting the coefficients until convergence
    #       based on the specified tolerance (`tol`).
    #     - If the difference between the old and new coefficients becomes smaller than the tolerance, the iteration stops early.
    #     - Negative values in the baseline-corrected signal are set to zero to avoid unrealistic baseline values.
    #     """
    #     # original_y = y.copy()
    #     # order = deg + 1
    #     # coeffs = np.ones(order)
    #     # cond = math.pow(abs(y).max(), 1.0 / order)
    #     # x = np.linspace(0.0, cond, y.size)  # Ensure this generates the expected range
    #     # base = y.copy()
    #     # vander = np.vander(x, order)  # Could potentially generate huge matrix if misconfigured
    #     # vander_pinv = np.linalg.pinv(vander)
    #     # for _ in range(max_it):
    #     #     coeffs_new = np.dot(vander_pinv, y)
    #     #     if np.linalg.norm(coeffs_new - coeffs) / np.linalg.norm(coeffs) < tol:
    #     #         break
    #     #     coeffs = coeffs_new
    #     #     base = np.dot(vander, coeffs)
    #     #     y = np.minimum(y, base)
    #     baseline, corrected = self.snip_baseline(y)

    #     # Calculate maximum peak amplitude (3 x baseline amplitude)
    #     N = len(y)
    #     smooth_hw = max(5, min(N//20, 51))     # e.g., ~5% of series length
    #     interp_hw = max(5, min(N//10, 101))
    #     num_std = 2.5
    #     baseline_fitter = Baseline(x)
    #     # fit, params_mask = baseline_fitter.std_distribution(y, 45)#, smooth_half_window=10)
    #     fit, params_mask = baseline_fitter.std_distribution(y, smooth_half_window=smooth_hw, interp_half_window=interp_hw, num_std=num_std)
    #     mask = params_mask['mask'] #  Mask for regions of signal without peaks
    #     if not np.any(mask):
    #         # robust noise from first-difference
    #         dif = np.diff(y)
    #         mad = 1.4826 * np.median(np.abs(dif - np.median(dif))) / np.sqrt(2.0)
    #         if mad == 0 or not np.isfinite(mad):
    #             # last resort: take middle 60% as baseline-ish
    #             lo, hi = int(0.2*N), int(0.8*N)
    #             mask = np.zeros(N, bool); mask[lo:hi] = True
    #         else:
    #             # baseline-ish points are where |y - median| < k * mad
    #             k = 3.0
    #             mask = np.abs(y - np.median(y)) < k * mad
    #     # min_peak_amp = (y[mask].max()-y[mask].min())*3
    #     min_peak_amp = (np.std(y[mask]))*3 # 2 sigma times 3
    #     print(f"Minimum amplitude: {min_peak_amp}")
    #     # min_peak_amp = (base.max()-base.min())*3
    #     # min_peak_amp = np.std(original_y-base)*3
    #     return baseline, min_peak_amp # return base
    
    # def baseline(self, x, y, deg=5, max_it=1000, tol=1e-4):
    #     # Final SNIP with width-aware iterations
    #     w_pts = self._estimate_typical_width_points(x, y)
    #     b, c = self.snip_baseline(y, peak_width_points=w_pts, max_iter=None, pad_edges=True)
    
    #     # --- minimum peak amplitude ---
    #     c = np.clip(c, 0, None)
    #     sigma_amp = self._robust_sigma_amp(c)
    
    #     # k-sigma rule (defensible)
    #     k = 5.0
    #     t_k = k * sigma_amp
    
    #     # floors to prevent silly small thresholds
    #     dyn = np.nanpercentile(y, 99) - np.nanpercentile(y, 1)
    #     abs_floor = 0.005 * dyn                      # 0.5% of dynamic range
    #     rel_floor = 0.02 * (np.nanmedian(b) if np.isfinite(np.nanmedian(b)) else 0.0)  # 2% of baseline level
    
    #     min_peak_amp = max(t_k, abs_floor, rel_floor)
    #     print(min_peak_amp)
    #     min_peak_amp = min_peak_amp*3
    #     print(min_peak_amp)
    #     return b, float(min_peak_amp)

    def asls_baseline(self, y, lam=1e6, p=0.001, max_iter=50, conv_thresh=1e-6, return_info=True):
        """
        Asymmetric Least Squares baseline (Eilers & Boelens, 2005).
    
        Solves:  minimize_b  sum_i w_i (y_i - b_i)^2  +  lam * || D^2 b ||^2
        where weights w_i are updated asymmetrically:
            w_i = p         if (y_i > b_i)  (point above baseline; likely peak)
            w_i = 1 - p     otherwise        (point below baseline; baseline-ish)
    
        Parameters
        ----------
        y : (n,) array_like
            Input signal.
        lam : float
            Smoothness penalty (larger = smoother baseline). Typical LC: 1e6–1e8.
        p : float in (0, 0.5)
            Asymmetry parameter; smaller keeps baseline under peaks (0.001–0.01 common).
        max_iter : int
            Max IRLS iterations.
        conv_thresh : float
            Relative change threshold for convergence (||b_new - b|| / ||b||).
        return_info : bool
            If True, return (b, info_dict). Else return b only.
    
        Returns
        -------
        b : (n,) ndarray
            Estimated baseline.
        info : dict (optional)
            {'iterations': int, 'converged': bool, 'last_delta': float, 'weights': w}
        """
        y = np.asarray(y, dtype=float).copy()
        n = y.size
        if n < 3:
            b = np.maximum(y, 0.0)
            return (b, {'iterations': 0, 'converged': True, 'last_delta': 0.0, 'weights': np.ones_like(y)}) if return_info else b
    
        # Handle NaNs (linear interpolate)
        nan_mask = ~np.isfinite(y)
        if nan_mask.any():
            xi = np.arange(n)
            y[ nan_mask] = np.interp(xi[nan_mask], xi[~nan_mask], y[~nan_mask])
    
        # 2nd-difference operator D: shape (n-2, n)
        # D @ b ~ [b0 - 2b1 + b2, b1 - 2b2 + b3, ..., b_{n-3} - 2b_{n-2} + b_{n-1}]
        diagonals = [np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)]
        offsets   = [0, 1, 2]
        D = sparse.diags(diagonals, offsets, shape=(n-2, n), format='csc')
    
        # Penalty matrix L = D^T D (symmetric pentadiagonal)
        L = (D.T @ D).tocsc()
    
        # IRLS (iteratively reweighted least squares)
        w = np.ones(n)
        b = y.copy()
    
        for it in range(1, max_iter + 1):
            # Build W and normal equations: (W + lam*L) b = W y
            W = sparse.diags(w, 0, shape=(n, n), format='csc')
            A = W + lam * L
            rhs = w * y
    
            # Solve
            b_new = spsolve(A, rhs)
    
            # Update weights (asymmetry): small epsilon to avoid exact zeros
            r = y - b_new
            w = p * (r > 0.0) + (1.0 - p) * (r <= 0.0)
            w = np.clip(w, 1e-6, 1.0)
    
            # Convergence check
            denom = np.linalg.norm(b) + 1e-12
            delta = np.linalg.norm(b_new - b) / denom
            b = b_new
            if delta < conv_thresh:
                info = {'iterations': it, 'converged': True, 'last_delta': float(delta), 'weights': w}
                return (np.maximum(b, 0.0), info) if return_info else np.maximum(b, 0.0)
    
        info = {'iterations': max_iter, 'converged': False, 'last_delta': float(delta), 'weights': w}
        return (np.maximum(b, 0.0), info) if return_info else np.maximum(b, 0.0)
    
    def asls(self, y, lam=1e6, p=0.002, max_iter=50):
        b, info = self.asls_baseline(y, lam=lam, p=p, max_iter=max_iter, conv_thresh=1e-6, return_info=True)
        return b, info
    
    def baseline(self, x, y, *args, **kwargs):
        y = np.asarray(y, float)
    
        # choose lam and p from data
        dx = np.median(np.diff(np.asarray(x, float))) if len(x) > 1 else 1.0
        span = (x.max() - x.min()) if len(x) else 1.0
    
        lam = 1e6 * max(1.0, (span / max(dx, 1e-6)) / 200.0)  
        p   = 0.01                                          
    
        b, info = self.asls(y, lam=lam, p=p, max_iter=50)
        b = np.maximum(b, 0.0)
        c = np.clip(y - b, 0, None)
    
        # robust noise on c for a defensible detection floor
        d = np.diff(c)
        mad = 1.4826 * np.median(np.abs(d - np.median(d)))
        sigma = (mad / np.sqrt(2.0)) if (mad > 0 and np.isfinite(mad)) else 0.0
        k = 5.0
        dyn = np.nanpercentile(y, 99) - np.nanpercentile(y, 1)
        abs_floor = 0.005 * dyn
        rel_floor = 0.02 * np.nanmedian(b) if np.isfinite(np.nanmedian(b)) else 0.0
        min_peak_amp = max(k * sigma, abs_floor, rel_floor)
        return b, float(min_peak_amp*3)
    ######################################################
    #################  Peak Select  ######################
    ######################################################

    def highlight_subplot(self):
        """
        Highlights the current subplot by changing its border color to red, while resetting all other subplots' borders to default.

        Returns
        -------
        None
            This function modifies the subplot borders in place and updates the plot display.

        Notes
        -----
        - The function first resets all subplot borders to black.
        - The current subplot, identified by `self.current_ax_idx`, is then highlighted with a red border.
        - The `plt.sca(current_ax)` call ensures that the current axes are set to the highlighted subplot for further plotting operations.
        - The plot is redrawn using `plt.draw()` to reflect the changes visually.
        """
        # Reset all subplot borders to default (none or black)
        for ax in self.axs:
            ax.spines["top"].set_color("k")
            ax.spines["bottom"].set_color("k")
            ax.spines["left"].set_color("k")
            ax.spines["right"].set_color("k")

        # Highlight the current subplot with a red border
        current_ax = self.axs[self.current_ax_idx]
        current_ax.spines["top"].set_color("red")
        current_ax.spines["bottom"].set_color("red")
        current_ax.spines["left"].set_color("red")
        current_ax.spines["right"].set_color("red")
        plt.sca(current_ax)  # Set the current Axes instance to current_ax
        plt.draw()

    def on_click(self, event):
        """
        Handles mouse click events within the plot area to select peaks or mark positions where no peak is found.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event object containing information about the click, including the x and y coordinates,
            the axis in which the click occurred, and other metadata.

        Returns
        -------
        None
            This function updates the plot and internal data structures based on the click action.

        Notes
        -----
        - If the click occurs within a plot axis (`event.inaxes`), the function retrieves the corresponding trace and dataset.
        - The function checks if the click is close to a detected peak (within a threshold of 0.15). If a peak is found, it calls `handle_peak_selection` to process the peak.
        - If no peak is found near the click position, a vertical line and text annotation are added at the click location, marking the position as "No peak."
        - The click actions, such as selecting a peak or adding a line, are stored in `self.action_stack` for undo functionality.
        - Updates to the plot are redrawn using `plt.draw()` after each action.
        """
        self.x_full = []
        self.y_full = []
        if event.inaxes not in self.axs_to_traces:
            return
        # print("Click registered!")
        ax = event.inaxes
        # Assuming axs_to_traces maps axes to trace identifiers directly
        trace = self.axs_to_traces[ax]
        ax_idx = list(ax.figure.axes).index(ax)  # Retrieve the index of ax in the figure's list of axes

        xdata, y_bcorr = self.datasets[ax_idx]
        self.x_full = xdata
        self.y_full = y_bcorr
        peaks = self.peaks_indices[ax_idx]
        for i in peaks:
            plt.draw()
        rel_click_pos = np.abs(xdata[peaks] - event.xdata)
        peak_found = False
        for peak_index, peak_pos in enumerate(rel_click_pos):
            if peak_pos < 0.1:  # Threshold to consider a click close enough to a peak
                peak_found = True
                selected_peak = peaks[np.argmin(np.abs(xdata[peaks] - event.xdata))]
                # Correctly pass the trace identifier to handle_peak_selection
                self.handle_peak_selection(ax, ax_idx, xdata, y_bcorr, selected_peak, peaks, trace)
                # Store the action for undoing
                self.action_stack.append(("select_peak", ax, (ax_idx, selected_peak)))
                if self.cheers:
                    self.nice()
                break
        if not peak_found:
            self._nopeak_id += 1
            no_peak_key = (ax_idx, f"nopeak-{self._nopeak_id}")
            line = ax.axvline(event.xdata, color="grey", linestyle="--", zorder=-1)
            text = ax.text(event.xdata + 2, (ax.get_ylim()[1] / 10) * 0.7, "No peak\n" + str(np.round(event.xdata)), color="grey", fontsize=8)
            self.no_peak_lines[no_peak_key] = (line, text)
            self.integrated_peaks[no_peak_key] = {"area": 0, "rt": event.xdata, "text": text, "line": [line], "trace": trace, "area_ensemble": 0}
            self.action_stack.append(("add_nopeak", ax, no_peak_key))
            if self.cheers:
                self.oof()
            plt.grid(False)
            plt.draw()

    def auto_select_peaks(self):
        """
        Automatically selects peaks based on reference retention times for each compound in the dataset.

        Returns
        -------
        None
            This function updates the plot and internal data structures based on the reference peak positions.

        Notes
        -----
        - The function iterates through the `self.reference_peaks` dictionary, where each compound is associated with a list of reference retention times (RTs).
        - For each compound and trace, it checks if the compound is present in the `GDGT_dict` for that trace.
        - If a trace matches, the function attempts to find a peak close to the reference retention time.
        - If a peak is found within a threshold of 0.2 minutes from the reference retention time, the peak is selected using `handle_peak_selection`.
        - If no peak is found within the threshold, a vertical red line and text annotation ("No peak") are added to the plot at the reference retention time.
        - The click actions, such as selecting a peak or adding a line, are stored in `self.action_stack` for undo functionality.
        - The plot is updated and redrawn using `plt.draw()` after each action.
        """
        self.x_full = []
        self.y_full = []

        if self.reference_peaks:
            for compound, ref_peaks in self.reference_peaks.items():
                # Here, compound corresponds to the GDGT compound name (e.g., 'IIIa', 'IIb')
                for trace_id in self.traces:
                    if trace_id in self.GDGT_dict and compound in self.GDGT_dict[trace_id]:
                        ax_idx = self.traces.index(trace_id) if trace_id in self.traces else -1
                        if ax_idx != -1:
                            ax = self.axs[ax_idx]
                            xdata, y_bcorr = self.datasets[ax_idx]
                            self.x_full = xdata
                            self.y_full = y_bcorr
                            peaks = self.peaks_indices[ax_idx]
                            for ref_peak in ref_peaks["rts"]:
                                rel_click_pos = np.abs(xdata[peaks] - ref_peak)
                                peak_found = False
                                trace = self.axs_to_traces[self.axs[ax_idx]]
                                for peak_index, peak_pos in enumerate(rel_click_pos):
                                    if np.min(np.abs(xdata[peaks] - ref_peak)) < 0.2:  # Slightly higher threshold
                                        peak_found = True
                                        selected_peak = peaks[np.argmin(np.abs(xdata[peaks] - ref_peak))]
                                        self.handle_peak_selection(ax, ax_idx, xdata, y_bcorr, selected_peak, peaks, trace)
                                        if self.cheers:
                                            self.nice()
                                        break
                                if not peak_found:
                                    self._nopeak_id += 1
                                    no_peak_key = (ax_idx, f"nopeak-{self._nopeak_id}")
                                    line = ax.axvline(ref_peak, color="red", linestyle="--", alpha=0.5)
                                    text = ax.text(ref_peak + 2, ax.get_ylim()[1] * 0.5, "No peak\n" + str(np.round(ref_peak)), color="grey", fontsize=8)
                                    self.no_peak_lines[no_peak_key] = (line, text)
                                    self.integrated_peaks[no_peak_key] = {"area": 0, "rt": ref_peak, "trace": trace, "area_ensemble": 0}
                                    self.action_stack.append(("add_line", ax, no_peak_key))
                                    if self.cheers:
                                        self.oof()
                                    plt.draw()

    def on_key(self, event):
        """
        Handles keyboard input events for controlling the peak selection and plot interactions.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The key event object containing information about the key pressed and the current state of the plot.

        Returns
        -------
        None
            This function performs actions based on the key pressed and updates internal states or the plot accordingly.

        Notes
        -----
        - "Enter": Calls `collect_peak_data()` to finalize peak selection and closes the figure to resume script execution.
        - "d": Calls `undo_last_action()` to undo the most recent action.
        - "e": Reserved for future expansion (currently does nothing).
        - "up" and "down": Navigates between subplots using the up and down arrow keys, and highlights the selected subplot.
        - "r": Clears the peaks in the currently highlighted subplot by calling `clear_peaks_subplot()` and removes corresponding entries from `self.integrated_peaks` and `self.peak_results`.
        - After clearing or navigating, the plot is redrawn using `plt.draw()` to reflect changes.
        """
        if event.key == "enter":
            self.collect_peak_data()
            self.waiting_for_input = False
            plt.close(self.fig)  # Close the figure to resume script execution
        elif event.key == "d":
            self.undo_last_action()
        elif event.key in ["up", "down"]:
            # Handle subplot navigation with up and down arrow keys
            if event.key == "up":
                self.current_ax_idx = (self.current_ax_idx - 1) % len(self.axs)
            elif event.key == "down":
                self.current_ax_idx = (self.current_ax_idx + 1) % len(self.axs)
            self.highlight_subplot()
        elif event.key == "r":
            self.clear_peaks_subplot(self.current_ax_idx)
            trace_to_clear = self.axs_to_traces[self.axs[self.current_ax_idx]]

            # Remove any entries in self.integrated_peaks that have a matching trace value
            self.integrated_peaks = {key: peak_data for key, peak_data in self.integrated_peaks.items() if "trace" in peak_data and peak_data["trace"] != trace_to_clear}

            # Clear the corresponding entries in self.peak_results
            if trace_to_clear in self.peak_results:
                self.peak_results[trace_to_clear]["rts"] = []
                self.peak_results[trace_to_clear]["areas"] = []
            plt.draw()
        elif event.key == "t":
            print(f"All peaks removed from {self.sample_name}. Reference peaks will be updated.")
            self.clear_all_peaks()
            self.t_pressed = True
        elif event.key == "w":
            print("A new view!")
            self.add_window_controls()

    # def undo_last_action(self):
    #     """
    #     Undoes the last action performed during peak selection, removing the corresponding graphical objects from the plot.

    #     Returns
    #     -------
    #     None
    #         This function updates the plot and internal data structures by undoing the last peak-related action.

    #     Notes
    #     -----
    #     - The function checks the `self.action_stack` for the most recent action and removes the corresponding graphical objects
    #       (lines, fills, text annotations) from the plot.
    #     - If a valid peak is found in `self.integrated_peaks`, its graphical components (line, fill, text) are removed.
    #     - If no graphical components are found for the given key, a message is printed.
    #     - If no actions are available to undo, a message indicating "No actions to undo" is printed.
    #     - The plot is redrawn using `plt.draw()` after the graphical components are removed.
    #     """
    #     if self.action_stack:
    #         last_action, ax, key = self.action_stack.pop()
    #         peak_data = self.integrated_peaks.pop(key, None)
    #         if peak_data:
    #             if "line" in peak_data:
    #                 for line in peak_data["line"]:
    #                     line.remove()
    #             if "fill" in peak_data:
    #                 peak_data["fill"].remove()
    #             if "text" in peak_data:
    #                 peak_data["text"].remove()
    #             plt.draw()
    #         else:
    #             print(f"No graphical objects found for key {key}, action: {last_action}")
    #     else:
    #         print("No actions to undo.")
    def undo_last_action(self):
        if not self.action_stack:
            print("No actions to undo.")
            return
    
        last_action, ax, key = self.action_stack.pop()
    
        if last_action == "add_nopeak":
            # Remove the specific grey line & label by its unique key
            line, text = self.no_peak_lines.pop(key, (None, None))
            if line is not None:
                line.remove()
            if text is not None:
                text.remove()
            # If you mirrored it in integrated_peaks, clean that too:
            self.integrated_peaks.pop(key, None)
            plt.draw()
            return
    
        # existing peak-selection undo path (kept as-is)
        peak_data = self.integrated_peaks.pop(key, None)
        if peak_data:
            if "line" in peak_data:
                for ln in peak_data["line"]:
                    ln.remove()
            if "fill" in peak_data and peak_data["fill"] is not None:
                peak_data["fill"].remove()
            if "text" in peak_data and peak_data["text"] is not None:
                peak_data["text"].remove()
            plt.draw()
        else:
            print(f"No graphical objects found for key {key}, action: {last_action}")

    def clear_peaks_subplot(self, ax_idx):
        """
        Clears the peaks and resets the specified subplot by re-plotting the data.

        Parameters
        ----------
        ax_idx : int
            The index of the subplot (axis) to be cleared and reset.

        Returns
        -------
        None
            This function modifies the specified subplot in place and redraws the plot.

        Notes
        -----
        - The function clears the selected subplot using `ax.clear()`.
        - After clearing, it re-initializes the subplot by calling `setup_subplot()` to re-plot the data.
        - The plot is updated and redrawn using `plt.draw()` to reflect the changes.
        """
        ax = self.axs[ax_idx]
        ax.clear()
        self.setup_subplot(ax, ax_idx)
        plt.draw()
    def clear_all_peaks(self):
        """
        Clears all peaks and resets all subplots by re-plotting the data.

        This method iterates through each subplot, clears the peaks, and removes corresponding
        entries from the internal data structures `self.integrated_peaks` and `self.peak_results`.

        Returns
        -------
        None
        """
        for ax_idx in range(len(self.axs)):
            # Clear peaks for each subplot
            self.clear_peaks_subplot(ax_idx)
            trace_to_clear = self.axs_to_traces[self.axs[ax_idx]]

            # Remove any entries in self.integrated_peaks that have a matching trace value
            keys_to_remove = [key for key, peak_data in self.integrated_peaks.items() if "trace" in peak_data and peak_data["trace"] == trace_to_clear]
            for key in keys_to_remove:
                del self.integrated_peaks[key]

            # Clear the corresponding entries in self.peak_results
            if trace_to_clear in self.peak_results:
                self.peak_results[trace_to_clear]["rts"] = []
                self.peak_results[trace_to_clear]["areas"] = []

        # Clear the action stack since all actions are undone
        self.action_stack.clear()

        # Redraw the plot to reflect changes
        plt.draw()
    def collect_peak_data(self):
        """
        Collects and organizes peak data based on the GDGT (Glycerol Dialkyl Glycerol Tetraether) type provided.

        Returns
        -------
        None
            This function updates the `self.peak_results` dictionary with peak data for each trace.
        Notes
        -----
        - The function retrieves the appropriate GDGT dictionary (`self.GDGT_dict`) to determine the compounds for each trace.
        - It then collects peaks from `self.integrated_peaks` that match each trace and organizes them by retention time (RT).
        - If multiple compounds are associated with a trace, the function assigns peaks to compounds based on their order in the list. If fewer peaks are found than expected, a warning is issued.
        - For traces that correspond to a single compound, the first peak is selected and added to the results.
        - The `_append_peak_data` method is used to store the peak data for each compound in the `self.peak_results` dictionary.
        - Warnings are printed if no peaks or fewer peaks than expected are found for a given trace.
        """
        self.peak_results = {}

        # Get the correct GDGT dictionary
        gdgt_dict = self.GDGT_dict
        for trace_key, compounds in gdgt_dict.items():
            # Find matching peaks in self.integrated_peaks
            matching_peaks = [peak_data for key, peak_data in self.integrated_peaks.items() if peak_data["trace"] == trace_key]
            matching_peaks.sort(key=lambda peak: peak["rt"])
            if isinstance(compounds, list):  # If the key maps to multiple compounds
                if len(matching_peaks) < len(compounds):
                    print(f"Warning: Fewer peaks found than expected for trace {trace_key}")
                for i, compound in enumerate(compounds):
                    if i < len(matching_peaks):
                        self._append_peak_data(compound, matching_peaks[i])
                    else:
                        print(f"Warning: Not enough peaks to match all compounds for trace {trace_key}")
            else:  # Single compound
                if matching_peaks:
                    self._append_peak_data(compounds, matching_peaks[0])
                else:
                    print(f"Warning: No peaks found for trace {trace_key}")

    def _append_peak_data(self, compound, peak_data):
        """
        Helper function to append peak data to the `peak_results` dictionary.

        Parameters
        ----------
        compound : str
            The name of the compound (e.g., GDGT type) for which peak data is being stored.
        peak_data : dict
            A dictionary containing peak data, which includes the area under the peak and the retention time (rt).
            Example: {"area": float, "rt": float}

        Returns
        -------
        None
            This function updates the `self.peak_results` dictionary by appending the peak area and retention time for the given compound.

        Notes
        -----
        - If the compound is not already in `self.peak_results`, a new entry is created with empty lists for "areas" and "rts".
        - The peak area and retention time (rt) are appended to the corresponding lists for the given compound.
        """
        if compound not in self.peak_results:
            self.peak_results[compound] = {"areas": [], "rts": [], "area_ensemble": []}
        # print("peak_results", self.peak_results)
        # print("peak_data", peak_data)
        self.peak_results[compound]["areas"].append(peak_data["area"])
        self.peak_results[compound]["rts"].append(peak_data["rt"])
        self.peak_results[compound]["area_ensemble"].append(peak_data["area_ensemble"])

    def nice(self):
        import simpleaudio as sa
        import os
        here = os.path.dirname(__file__)
        wav_path = os.path.join(here, "nice.wav")
        wave_obj = sa.WaveObject.from_wave_file(wav_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    def oof(self):
        import simpleaudio as sa
        import os
        here = os.path.dirname(__file__)
        wav_path = os.path.join(here, "oof.wav")
        wave_obj = sa.WaveObject.from_wave_file(wav_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()