"""
Created on Tue Apr 23 23:59:05 2024
A translation of origami, developed by Jess Tierney in matlab,
and modified for GDGT chromatography. Developed with
Dr. Elizbaeth Thomas, OSIBL lab group at the University
at Buffalo.

Input: path to the sample .csv files output from openChrom.

Output: An output folder in the user-defined path (input)
        containing a .csv file with GDGT peak areas and a
        subfolder with sample-specific chromatograms that
        include integration area.

Instructions: provide the path to the directory containing
        the .csv files from openChrom and select the peaks
        in the chromatogram. Any peaks can be picked with
        a left click. The code expects the correct number
        of peaks for each trace (e.g., 5- and 6-methyl).
        Left click to pick the peak. "d" key to delete the
        last picked peak. Hit enter once finished selecting
        to move onto the next traceset/sample. If a sample
        needs to be rerun, go into the results output and delete
        the row with the sample name. The code will identify
        that the sample is not in the output and include it in
        the run. The output is saved after each sample so
        processing does not need to be done at one time.


@author: ~GAO~
@date: 11092024

"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.sparse import eye, diags, csc_matrix
from scipy.sparse.linalg import factorized
import warnings


class GDGTAnalyzer:
    def __init__(self, df, traces, window_bounds, GDGT_dict, gaus_iterations, sample_name, is_reference, max_peaks, sw, sf, pk_sns, pk_pr, reference_peaks=None):
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
        self.peaks = {}  # Store all peak indices and properties for each trace
        self.axs_to_traces = {}  # Empty map for connecting traces to figure axes
        self.peak_results = {}
        self.gi = gaus_iterations
        self.max_peaks_for_neighborhood = max_peaks
        self.peak_properties = {}
        self.smoothing_params = [sw, sf]
        self.pk_sns = pk_sns
        self.pk_pr = pk_pr

    def run(self):
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

        return self.peak_results, self.fig, self.reference_peaks

    ######################################################
    ################  Gaussian Fit  ######################
    ######################################################
    def multigaussian(self, x, *params):
        """Combined Gaussian function for multiple peaks"""
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp = params[i]
            cen = params[i + 1]
            wid = params[i + 2]
            y += amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))
        return y

    def gaussian(self, x, amp, cen, wid, dec):
        """Gaussian function for a single peak with a decay parameter."""
        return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2)) * np.exp(-dec * abs(x - cen))

    def individual_gaussian(self, x, amp, cen, wid):
        """Single Gaussian function"""
        return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))

    def estimate_initial_gaussian_params(self, x, y, peak):
        # Subset peaks so that only idx positions with x bounds are considered
        heights = []
        means = []
        stddevs = []

        height = y[peak]
        mean = x[peak]
        half_max = 0.5 * height  # change to 0.5

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
        """Find valleys based on the lowest points between peaks"""
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

    def find_peak_neighborhood_boundaries(self, x, y_smooth, peaks, valleys, peak_idx, ax, max_peaks, trace):
        peak_distances = np.abs(x[peaks] - x[peak_idx])
        closest_peaks_indices = np.argsort(peak_distances)[:max_peaks]
        closest_peaks = np.sort(peaks[closest_peaks_indices])

        overlapping_peaks = []
        extended_boundaries = {}
        # Analyze each of the closest peaks
        for peak in closest_peaks:
            peak_pos = np.where(peak == peaks)
            l_lim = self.peak_properties[trace]["left_bases"][peak_pos][0]
            r_lim = self.peak_properties[trace]["right_bases"][peak_pos][0]
            heights, means, stddevs = self.estimate_initial_gaussian_params(x[l_lim : r_lim + 1], y_smooth[l_lim : r_lim + 1], peak)
            height, mean, stddev = heights[0], means[0], stddevs[0]

            # Fit Gaussian and get best fit parameters
            popt, _ = curve_fit(self.individual_gaussian, x, y_smooth, p0=[height, mean, stddev], maxfev=self.gi)
            # popt, _ = curve_fit(self.gaussian, x, y_smooth, p0=[height, mean, stddev, 0.1], maxfev=self.gi)
            # Extend Gaussian fit limits
            x_min, x_max = self.calculate_gaus_extension_limits(popt[1], popt[2], 0, factor=3)
            extended_x, extended_y = self.extrapolate_gaussian(x, popt[0], popt[1], popt[2], None, x_min - 1, x_max + 1)
            # Find the boundaries based on the derivative test
            peak_x_value = x[peak]
            n_peak_idx = np.argmin(np.abs(extended_x - peak_x_value))
            left_idx, right_idx = self.calculate_boundaries(extended_x, extended_y, n_peak_idx)
            extended_boundaries[peak] = (extended_x[left_idx], extended_x[right_idx])

        # Determine the peak of interest boundaries
        poi_bounds = extended_boundaries.get(peak_idx, (None, None))

        # Check for overlaps and determine the neighborhood
        for peak, bounds in extended_boundaries.items():
            if peak < peak_idx and bounds[1] > poi_bounds[0]:  # Overlaps to the left
                overlapping_peaks.append(peak)
            elif peak > peak_idx and bounds[0] < poi_bounds[1]:  # Overlaps to the right
                overlapping_peaks.append(peak)

        # Calculate neighborhood boundaries based on the left-most and right-most overlapping peaks
        if overlapping_peaks:
            left_most_peak = min(overlapping_peaks, key=lambda p: extended_boundaries[p][0])
            right_most_peak = max(overlapping_peaks, key=lambda p: extended_boundaries[p][1])
            neighborhood_left_boundary = extended_boundaries[left_most_peak][0]
            neighborhood_right_boundary = extended_boundaries[right_most_peak][1]
        else:
            # Use the peak of interest's bounds if no other peaks are overlapping
            neighborhood_left_boundary = poi_bounds[0]
            neighborhood_right_boundary = poi_bounds[1]
        return neighborhood_left_boundary, neighborhood_right_boundary, overlapping_peaks

    def calculate_boundaries(self, x, y, ind_peak):
        """
        Function to find boundaries of a single peak based on first first derivative test.
        Input: retention time (x) and signal (y) data. Index position of the peak of interst (ind_peak).
        Output: Two integer values, denoting the index position of the left (A) and right (B) boundaries of the peak.
        """
        smooth_y = self.smoother(y)
        velocity, X1 = self.forward_derivative(x, smooth_y)
        velocity /= np.max(np.abs(velocity))
        smooth_velo = self.smoother(velocity)
        dt = int(np.ceil(0.025 / np.mean(np.diff(x))))
        A = np.where(smooth_velo[: ind_peak - 3 * dt] < self.pk_sns)[0]  # 0.05)[0]
        B = np.where(smooth_velo[ind_peak + 3 * dt :] > -self.pk_sns)[0]  # -0.05)[0]

        if A.size > 0:
            A = A[-1] + 1
        else:
            A = 1
        if B.size > 0:
            B = B[0] + ind_peak + 3 * dt - 1
        else:
            B = len(x) - 1
        return A, B

    # def calculate_boundaries(self, x, y, ind_peak):
    #     """
    #     Function to find boundaries of a single peak based on the second derivative test.
    #     Input: retention time (x) and signal (y) data. Index position of the peak of interest (ind_peak).
    #     Output: Two integer values, denoting the index position of the left (A) and right (B) boundaries of the peak.
    #     """
    #     # Smooth the y-values to reduce noise
    #     smooth_y = self.smoother(y)

    #     # Calculate the first derivative (velocity)
    #     velocity, X1 = self.forward_derivative(x, smooth_y)

    #     # Now calculate the second derivative (curvature)
    #     second_derivative = np.diff(velocity) / np.diff(X1)

    #     # Normalize the second derivative for stability
    #     second_derivative /= np.max(np.abs(second_derivative))

    #     # Smooth the second derivative to reduce noise
    #     smooth_second_deriv = second_derivative  # self.smoother(second_derivative)

    #     # Ensure dt is at least 1 (based on x step sizes)
    #     dt = max(int(np.ceil(0.025 / np.mean(np.diff(x)))), 1)

    #     # Find the left boundary (where second derivative is positive)
    #     A = np.where(smooth_second_deriv[: ind_peak - 3 * dt] > self.pk_sns)[0]  # Adjust threshold as necessary

    #     # Find the right boundary (where second derivative is negative)
    #     B = np.where(smooth_second_deriv[ind_peak + 3 * dt :] < -self.pk_sns)[0]  # originall 0.01 # CHANGED FROM ...:]<-self.pk_sns LIKELY CHANGE BACK

    #     # Set the left boundary
    #     if A.size > 0:
    #         A = A[-1] + 1
    #     else:
    #         A = 1

    #     # Set the right boundary
    #     if B.size > 0:
    #         B = B[0] + ind_peak + 3 * dt - 1
    #     else:
    #         B = len(x) - 1

    #     return A, B

    def find_peak_boundaries(self, x, y, center, trace, threshold=0.1):  # 0.005):
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

    def smoother(self, y, lambd=1):
        # """

        # Parameters
        # ----------
        # y : TYPE
        #     DESCRIPTION.
        # lambd : TYPE, optional
        #     DESCRIPTION. The default is 1.
        # Returns
        # -------
        # TYPE
        #     DESCRIPTION.

        # """
        # # # Convert pandas Series to numpy array if necessary
        # if isinstance(y, pd.Series):
        #     y = y.to_numpy()  # More modern way to convert to numpy array

        # m = len(y)
        # I = eye(m, format="csc")  # Create an identity matrix in CSC format
        # D = diags([1, -1], [0, 1], shape=(m - 1, m), format="csc")  # Create D matrix in CSC format

        # # Factorize the matrix once if the lambda value and the size of y do not change frequently
        # A = I + lambd * (D.T @ D)  # Operations with CSC formatted matrices
        # A = csc_matrix(A)  # Ensure A is in CSC format, though it should already be

        # solve = factorized(A)  # Pre-factorize the matrix to speed up solving
        # return solve(y)
        return savgol_filter(y, self.smoothing_params[0], self.smoothing_params[1])

    def forward_derivative(self, x, y):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        FD1 : TYPE
            DESCRIPTION.
        X1 : TYPE
            DESCRIPTION.

        """
        FD1 = np.diff(y) / np.diff(x)
        X1 = x[:-1]
        return FD1, X1

    def extrapolate_gaussian(self, x, amp, cen, wid, decay, x_min, x_max, step=0.1):
        """Extend Gaussian tails from x_min to x_max with given step."""
        extended_x = np.arange(x_min, x_max, step)
        if decay is None:
            extended_y = self.individual_gaussian(extended_x, amp, cen, wid)
        else:
            extended_y = self.gaussian(extended_x, amp, cen, wid, decay)
        return extended_x, extended_y

    def calculate_gaus_extension_limits(self, cen, wid, decay, factor=3):  # decay, factor=3):
        """Calculate the limits to which extend the Gaussian tails based on the 3-sigma rule."""
        sigma_effective = wid * factor  # Adjust factor for tail thinness
        extension_factor = 1 / decay if decay != 0 else sigma_effective  # Use decay to modify the extension if applicable
        x_min = cen - sigma_effective - np.abs(extension_factor)
        x_max = cen + sigma_effective + np.abs(extension_factor)
        return x_min, x_max

    def fit_gaussians(self, x_full, y_full, ind_peak, trace, peaks, ax):
        # detect overlapping peaks
        current_peaks = np.array(peaks)
        current_peaks = np.append(current_peaks, ind_peak)
        current_peaks = np.sort(current_peaks)
        iteration = 0
        best_fit_y = None
        best_x = None
        best_fit_params = None
        best_ksp = np.inf
        multi_gauss_flag = True
        best_idx_interest = None
        best_error = np.inf
        best_ks_stat = np.inf
        while len(current_peaks) > 1:
            left_boundary, _ = self.calculate_boundaries(x_full, y_full, np.min(current_peaks))
            _, right_boundary = self.calculate_boundaries(x_full, y_full, np.max(current_peaks))
            x = x_full[left_boundary : right_boundary + 1]
            y = y_full[left_boundary : right_boundary + 1]
            index_of_interest = np.where(current_peaks == ind_peak)[0][0]
            initial_guesses = []
            bounds_lower = []
            bounds_upper = []
            for peak in current_peaks:
                height, center, width = self.estimate_initial_gaussian_params(x, y, peak)  # peak)
                height = height[0]
                center = center[0]
                width = width[0]
                initial_guesses.extend([height, center, width])
                # Bounds for peak fitting
                lw = 0.1 - width if width > 0.1 else width
                bounds_lower.extend([0.1 * y_full[peak], x_full[peak] - 0.15, lw])  # Bounds for peak fittin
                bounds_upper.extend([1 + y_full[peak], x_full[peak] + 0.15, 0.5 + width])  # Old amplitude was 2 * peak height, y_full[peak] * 2, width was 2+width
            bounds = (bounds_lower, bounds_upper)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, pcov = curve_fit(self.multigaussian, x, y, p0=initial_guesses, method="dogbox", bounds=bounds, maxfev=self.gi)  # , ftol=1e-4, xtol=1e-4)
                fitted_y = self.multigaussian(x, *popt)
                # ax.plot(x, fitted_y, c="fuchsia") # plots the multi gaussian curve
                error = np.sqrt(((fitted_y - y) ** 2).mean())  # RMSE
                if error < best_error:
                    best_error = error
                    best_fit_params = popt
                    best_fit_y = fitted_y
                    best_x = x
                    best_idx_interest = index_of_interest
            except RuntimeError:
                pass

            distances = np.abs(x[current_peaks] - x_full[ind_peak])
            if distances.size > 0:
                max_dist_idx = np.argmax(distances)
                current_peaks = np.delete(current_peaks, max_dist_idx)
            iteration += 1

        # Final fit with only the selected peak
        if len(current_peaks) == 1:
            left_boundary, right_boundary = self.calculate_boundaries(x_full, y_full, ind_peak)
            x = x_full[left_boundary : right_boundary + 1]
            y = y_full[left_boundary : right_boundary + 1]
            height, center, width = self.estimate_initial_gaussian_params(x, y, ind_peak)
            height = height[0]
            center = center[0]
            width = width[0]
            # p0 = [height, center, width]
            initial_decay = 0.1
            p0 = [height, center, width, initial_decay]
            bounds_lower = [0.9 * y_full[ind_peak], x_full[ind_peak] - 0.1, 0.5 * width, 0.01]  # modified width from 0.05
            bounds_upper = [1 + y_full[ind_peak], x_full[ind_peak] + 0.1, width * 1.5, 2]
            bounds = (bounds_lower, bounds_upper)
            try:
                # Initial try with given maxfev
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: self.gaussian(x, amp, cen, wid, dec), x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=self.gi)
                single_fitted_y = self.gaussian(x, *single_popt)
                error = np.sqrt(((single_fitted_y - y) ** 2).mean())  # RMSE
                if error < best_error:
                    multi_gauss_flag = False
                    best_error = error
                    best_fit_params = single_popt
                    best_fit_y = single_fitted_y
                    best_x = x
            except RuntimeError:
                print(f"Warning: Optimal parameters could not be found with {self.gi} iterations. Increasing iterations by a factor of 100. Please be patient.")

                # Increase maxfev by a factor of 10 and retry
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: self.gaussian(x, amp, cen, wid, dec), x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=self.gi * 100)
                    single_fitted_y = self.gaussian(x, *single_popt)
                    error = np.sqrt(((single_fitted_y - y) ** 2).mean())  # RMSE
                    if error < best_error:
                        multi_gauss_flag = False
                        best_error = error
                        best_fit_params = single_popt
                        best_fit_y = single_fitted_y
                        best_x = x
                except RuntimeError:
                    print("Error: Optimal parameters could not be found even after increasing the iterations.")
        if multi_gauss_flag == True:
            # print("picked multi", trace)
            # Determine the index of the peak of interest in the multi-Gaussian fit
            amp, cen, wid = best_fit_params[best_idx_interest * 3], best_fit_params[best_idx_interest * 3 + 1], best_fit_params[best_idx_interest * 3 + 2]
            best_fit_y = self.individual_gaussian(best_x, amp, cen, wid)
            best_x, best_fit_y = self.extrapolate_gaussian(best_x, amp, cen, wid, None, best_x.min() - 1, best_x.max() + 1, step=0.1)

            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            # ax.plot(best_x, best_fit_y, c="blue")

        else:
            # print("picked single", trace)
            x_min, x_max = self.calculate_gaus_extension_limits(best_fit_params[1], best_fit_params[2], best_fit_params[3], factor=3)
            best_x, best_fit_y = self.extrapolate_gaussian(best_x, best_fit_params[0], best_fit_params[1], best_fit_params[2], best_fit_params[3], x_min, x_max, step=0.1)
            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            # ax.plot(best_x, best_fit_y, c="fuchsia")
        area_smooth = simpson(y=best_fit_y, x=best_x)

        return best_x, best_fit_y, area_smooth

    def handle_peak_selection(self, ax, ax_idx, xdata, y_bcorr, peak_idx, peaks, trace):
        """


        Parameters
        ----------
        ax : TYPE
            DESCRIPTION.
        ax_idx : TYPE
            DESCRIPTION.
        xdata : TYPE
            DESCRIPTION.
        y_bcorr : TYPE
            DESCRIPTION.
        peak_idx : TYPE
            DESCRIPTION.
        peaks : TYPE
            DESCRIPTION.
        trace : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        try:
            valleys = self.find_valleys(y_bcorr, peaks)
            A, B, peak_neighborhood = self.find_peak_neighborhood_boundaries(xdata, y_bcorr, self.peaks[trace], valleys, peak_idx, ax, self.max_peaks_for_neighborhood, trace)
            x_fit, y_fit_smooth, area_smooth = self.fit_gaussians(xdata, y_bcorr, peak_idx, trace, peak_neighborhood, ax)
            fill = ax.fill_between(x_fit, 0, y_fit_smooth, color="grey", alpha=0.5)
            rt_of_peak = xdata[peak_idx]
            area_text = f"Area: {area_smooth:.0f}\nRT: {rt_of_peak:.0f}"
            text_annotation = ax.annotate(area_text, xy=(rt_of_peak + 1.5, y_fit_smooth.max() * 0.5), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="grey")
            self.integrated_peaks[(ax_idx, peak_idx)] = {"fill": fill, "area": area_smooth, "rt": rt_of_peak, "text": text_annotation, "trace": trace}
            plt.draw()
            # Update self.peak_results for the current trace
            ############# added to try and figure out whhy new peaks arent updated
            if trace not in self.peak_results:
                self.peak_results[trace] = {"rts": [], "areas": []}
            self.peak_results[trace]["rts"].append(rt_of_peak)
            self.peak_results[trace]["areas"].append(area_smooth)  # Calculate area if needed
            #############
        except RuntimeError:
            pass

    ######################################################
    ################      Plot      ######################
    ######################################################
    def plot_data(self):
        """

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        axs : TYPE
            DESCRIPTION.

        """
        # Create subplots
        if len(self.traces) == 1:
            fig, ax = plt.subplots(figsize=(8, 10))
            axs = [ax]  # Ensure axs is iterable even when there's only one plot
        else:
            fig, axs = plt.subplots(len(self.traces), 1, figsize=(8, 10), sharex=True)
            axs = axs.ravel()  # Ensure axs is iterable
        self.datasets = [None] * len(self.traces)
        self.peaks_indices = [None] * len(self.traces)
        for i, ax in enumerate(axs):
            self.setup_subplot(ax, i)
            if i == len(self.traces) - 1:
                ax.set_xlabel("Corrected Retention Time (minutes)")
        fig.suptitle(f"Sample: {self.sample_name}", fontsize=16, fontweight="bold")
        return fig, axs

    def setup_subplot(self, ax, trace_idx):
        x_values = self.df["rt_corr"]
        within_xlim = (x_values >= self.window_bounds[0]) & (x_values <= self.window_bounds[1])
        trace = self.traces[trace_idx]
        y = self.df[trace]
        trace = self.traces[trace_idx]
        # Baseline correction
        y_base = self.baseline(y)
        y_bcorr = y - y_base
        baseline = self.baseline(y_bcorr)
        if baseline < 1:
            baseline = 1
        y_bcorr[y_bcorr < 0] = 0
        y_filtered = self.smoother(y_bcorr)
        # Find peaks
        peaks_total, properties = find_peaks(y_filtered, height=np.mean(baseline) * 3, width=0.05, prominence=self.pk_pr)
        self.peaks[trace] = peaks_total  # Storing peaks and their properties
        self.peak_properties[trace] = properties
        ax.plot(self.df["rt_corr"], y_filtered, "k")
        ax.set_ylabel(f"Trace {trace}")
        ax.set_xlim(self.window_bounds)

        # x_values = self.df["rt_corr"]
        # within_xlim = (x_values >= self.window_bounds[0]) & (x_values <= self.window_bounds[1])
        y_within_xlim = y_filtered[within_xlim]
        if len(y_within_xlim) > 0:
            ymin, ymax = y_within_xlim.min(), y_within_xlim.max()
            y_margin = (ymax - ymin) * 0.1  # Add 10% margin to the top and bottom
            ax.set_ylim(0, ymax + y_margin)
        else:
            ax.set_ylim(0, 1)  # Default limits if no data within xlim

        # Store or update dataset and peak indices
        self.axs_to_traces[ax] = trace
        self.datasets[trace_idx] = (self.df["rt_corr"], y_bcorr)  # y_bcorr)
        self.peaks_indices[trace_idx] = peaks_total

    # def baseline(self, y, deg=400, max_it=1000, tol=1e-1):
    def baseline(self, y, deg=4, max_it=50, tol=1e-3):
        order = deg + 1
        coeffs = np.ones(order)
        cond = math.pow(abs(y).max(), 1.0 / order)
        x = np.linspace(0.0, cond, y.size)  # Ensure this generates the expected range
        base = y.copy()
        vander = np.vander(x, order)  # Could potentially generate huge matrix if misconfigured
        vander_pinv = np.linalg.pinv(vander)
        for _ in range(max_it):
            coeffs_new = np.dot(vander_pinv, y)
            if np.linalg.norm(coeffs_new - coeffs) / np.linalg.norm(coeffs) < tol:
                break
            coeffs = coeffs_new
            base = np.dot(vander, coeffs)
            y = np.minimum(y, base)
            y[y < 0] = 0
        return base

    ######################################################
    #################  Peak Select  ######################
    ######################################################

    def highlight_subplot(self):
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
        self.x_full = []
        self.y_full = []
        if event.inaxes:
            # print("Click registered!")
            ax = event.inaxes
            # Assuming axs_to_traces maps axes to trace identifiers directly
            trace = self.axs_to_traces[ax]
            ax_idx = list(ax.figure.axes).index(ax)  # Retrieve the index of ax in the figure's list of axes

            xdata, y_bcorr = self.datasets[ax_idx]
            self.x_full = xdata
            self.y_full = y_bcorr
            peaks = self.peaks_indices[ax_idx]
            rel_click_pos = np.abs(xdata[peaks] - event.xdata)
            peak_found = False
            for peak_index, peak_pos in enumerate(rel_click_pos):
                if peak_pos < 0.15:  # Threshold to consider a click close enough to a peak
                    peak_found = True
                    selected_peak = peaks[np.argmin(np.abs(xdata[peaks] - event.xdata))]
                    # Correctly pass the trace identifier to handle_peak_selection
                    self.handle_peak_selection(ax, ax_idx, xdata, y_bcorr, selected_peak, peaks, trace)
                    # Store the action for undoing
                    self.action_stack.append(("select_peak", ax, (ax_idx, selected_peak)))
                    break
            if not peak_found:
                peak_key = (ax_idx, None)  # Using ax_idx to keep consistent with non-peak-specific actions
                line = ax.axvline(event.xdata, color="grey", linestyle="--", zorder=-1)
                text = ax.text(event.xdata + 2, (ax.get_ylim()[1] / 10) * 0.7, "No peak\n" + str(np.round(event.xdata)), color="grey", fontsize=8)
                no_peak_key = peak_key
                self.no_peak_lines[no_peak_key] = (line, text)
                self.integrated_peaks[peak_key] = {"area": 0, "rt": event.xdata, "text": text, "line": [line], "trace": trace}
                self.action_stack.append(("add_line", ax, no_peak_key))
                plt.grid(False)
                plt.draw()

    def auto_select_peaks(self):
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
                                for peak_index, peak_pos in enumerate(rel_click_pos):
                                    if np.min(np.abs(xdata[peaks] - ref_peak)) < 0.2:  # Slightly higher threshold
                                        peak_found = True
                                        selected_peak = peaks[np.argmin(np.abs(xdata[peaks] - ref_peak))]
                                        # Use trace_id from axs_to_traces to pass to handle_peak_selection
                                        trace = self.axs_to_traces[self.axs[ax_idx]]
                                        self.handle_peak_selection(ax, ax_idx, xdata, y_bcorr, selected_peak, peaks, trace)
                                        break
                                if not peak_found:
                                    peak_key = (ax_idx, None)
                                    line = ax.axvline(ref_peak, color="red", linestyle="--", alpha=0.5)
                                    text = ax.text(ref_peak + 2, ax.get_ylim()[1] * 0.5, "No peak\n" + str(np.round(ref_peak)), color="grey", fontsize=8)
                                    no_peak_key = peak_key
                                    self.no_peak_lines[no_peak_key] = (line, text)
                                    self.integrated_peaks[peak_key] = {"area": 0, "rt": ref_peak, "trace": trace}
                                    self.action_stack.append(("add_line", ax, no_peak_key))
                                    plt.draw()

    def on_key(self, event):
        if event.key == "enter":
            self.collect_peak_data()
            self.waiting_for_input = False
            plt.close(self.fig)  # Close the figure to resume script execution
        elif event.key == "d":
            self.undo_last_action()
        elif event.key == "e":
            pass
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

    def undo_last_action(self):
        if self.action_stack:
            last_action, ax, key = self.action_stack.pop()
            peak_data = self.integrated_peaks.pop(key, None)
            if peak_data:
                if "line" in peak_data:
                    for line in peak_data["line"]:
                        line.remove()
                if "fill" in peak_data:
                    peak_data["fill"].remove()
                if "text" in peak_data:
                    peak_data["text"].remove()
                plt.draw()
            else:
                print(f"No graphical objects found for key {key}, action: {last_action}")
        else:
            print("No actions to undo.")

    def clear_peaks_subplot(self, ax_idx):
        ax = self.axs[ax_idx]
        ax.clear()
        self.setup_subplot(ax, ax_idx)
        plt.draw()

    def collect_peak_data(self):
        """
        Collects and organizes peak data based on the GDGT type provided.
        """
        self.peak_results = {}

        # Get the correct GDGT dictionary
        gdgt_dict = self.GDGT_dict

        for trace_key, compounds in gdgt_dict.items():
            # Find matching peaks in self.integrated_peaks
            matching_peaks = [peak_data for key, peak_data in self.integrated_peaks.items() if peak_data["trace"] == trace_key]
            # Sort peaks by retention time
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
        Helper function to append peak data to the peak_results dictionary.

        Parameters:
        - compound: str, the name of the compound.
        - peak_data: dict, the peak data including area and retention time.
        """
        if compound not in self.peak_results:
            self.peak_results[compound] = {"areas": [], "rts": []}

        self.peak_results[compound]["areas"].append(peak_data["area"])
        self.peak_results[compound]["rts"].append(peak_data["rt"])
