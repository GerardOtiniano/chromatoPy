# src/chromatopy/chromatoPy_base.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from pybaselines import Baseline
import warnings

class GDGTAnalyzer:
    def __init__(self, df, traces, window_bounds, GDGT_dict, gaus_iterations, sample_name, is_reference, max_peaks, sw, sf, pk_sns, pk_pr, max_PA, reference_peaks=None):
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

    def gaussian_decay(self, x, amp, cen, wid, dec):
        """
        Computes a Gaussian function with an added exponential decay term.
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
        dec : float
            Decay factor applied to the Gaussian to introduce exponential decay.
        Returns
        -------
        numpy.ndarray
            The values of the Gaussian function with exponential decay evaluated at each x-value.
        """
        return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2)) * np.exp(-dec * abs(x - cen))

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

    def find_peak_neighborhood_boundaries(self, x, y_smooth, peaks, valleys, peak_idx, ax, max_peaks, trace):
        """
        Finds the extended boundaries of a peak's neighborhood by analyzing the closest peaks and their overlaps.

        Parameters
        ----------
        x : numpy.ndarray or pandas.Series
            Array of x-values (e.g., retention times or wavelengths) where the peaks are located.
        y_smooth : numpy.ndarray or pandas.Series
            Array of smoothed y-values (e.g., intensity or absorbance) corresponding to the x-values.
        peaks : numpy.ndarray or list of int
            List of indices representing the positions of the detected peaks.
        valleys : numpy.ndarray or list of int
            List of indices representing the positions of the detected valleys between peaks.
        peak_idx : int
            Index of the peak of interest around which to find the neighborhood boundaries.
        ax : matplotlib.axes.Axes
            The axis object used for plotting (if needed).
        max_peaks : int
            Maximum number of nearby peaks to consider when determining the neighborhood.
        trace : str
            Identifier for the trace being analyzed (e.g., which sample or dataset the peaks belong to).

        Returns
        -------
        neighborhood_left_boundary : float
            The x-value of the left boundary of the peak's neighborhood.
        neighborhood_right_boundary : float
            The x-value of the right boundary of the peak's neighborhood.
        overlapping_peaks : list of int
            List of peaks that overlap with the peak of interest based on their extended boundaries.

        Notes
        -----
        - The function calculates Gaussian fits for nearby peaks to extend their boundaries and check for overlaps.
        - The boundaries of the peak of interest are determined based on its closest neighboring peaks and their overlaps.
        - If overlapping peaks are found, the neighborhood boundaries are adjusted accordingly.
        - If no peaks overlap, the neighborhood is confined to the bounds of the peak of interest.
        """
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
            try:
                popt, _ = curve_fit(self.individual_gaussian, x, y_smooth, p0=[height, mean, stddev], maxfev=self.gi)
            except RuntimeError:
                popt, _ = curve_fit(self.individual_gaussian, x, y_smooth, p0=[height, mean, stddev], maxfev=self.gi*100)
            # popt, _ = curve_fit(self.gaussian, x, y_smooth, p0=[height, mean, stddev, 0.1], maxfev=self.gi)
            # Extend Gaussian fit limits
            x_min, x_max = self.calculate_gaus_extension_limits(popt[1], popt[2], 0, factor=3)
            extended_x, extended_y = self.extrapolate_gaussian(x, popt[0], popt[1], popt[2], None, x_min - 2, x_max + 2)
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
         Calculates the left and right boundaries of a single peak based on the first derivative test.

         Parameters
         ----------
         x : numpy.ndarray or pandas.Series
             Array of x-values (e.g., retention times) where the peak and signal data are located.
         y : numpy.ndarray or pandas.Series
             Array of y-values (e.g., intensity or absorbance) corresponding to the x-values.
         ind_peak : int
             Index of the peak of interest in the x and y data for which the boundaries are to be determined.

         Returns
         -------
         A : int
             The index of the left boundary of the peak.
         B : int
             The index of the right boundary of the peak.

         Notes
         -----
         - This method uses the first derivative of the signal to detect the peak's boundaries.
         - The signal is smoothed before calculating the derivative to reduce noise.
         - The left boundary (`A`) is found by searching for the first point before the peak where the derivative
           is smaller than a specified sensitivity threshold (`self.pk_sns`).
         - The right boundary (`B`) is found by searching for the first point after the peak where the derivative
           becomes larger than the negative of the same sensitivity threshold.
         - If no left boundary is found, the function defaults to the start of the signal (index 1).
           If no right boundary is found, the function defaults to the end of the signal.
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
        return savgol_filter(y, param_0, param_1)

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

    def extrapolate_gaussian(self, x, amp, cen, wid, decay, x_min, x_max, step=0.01):
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

    def calculate_gaus_extension_limits(self, cen, wid, decay, factor=3):  # decay, factor=3):
        """
        Calculates the extension limits for the Gaussian tails based on the 3-sigma rule, optionally accounting for decay.

        Parameters
        ----------
        cen : float
            Center of the Gaussian function (peak position).
        wid : float
            Width of the Gaussian function (standard deviation of the distribution).
        decay : float
            Decay factor to modify the extension of the Gaussian tails. If decay is 0, it will be ignored.
        factor : float, optional
            Factor to extend the Gaussian tails, typically 3 to represent the 3-sigma rule. Default is 3.

        Returns
        -------
        x_min : float
            The lower limit (x-value) to which the Gaussian tail is extended.
        x_max : float
            The upper limit (x-value) to which the Gaussian tail is extended.

        Notes
        -----
        - The function uses the 3-sigma rule to calculate the limits for Gaussian tail extension, scaling by the `factor`.
        - If `decay` is non-zero, it modifies the extension limits based on the decay rate.
        - If `decay` is 0, the limits are calculated purely based on the Gaussian width and `factor`.
        """
        sigma_effective = wid * factor  # Adjust factor for tail thinness
        extension_factor = 1 / decay if decay != 0 else sigma_effective  # Use decay to modify the extension if applicable
        x_min = cen - sigma_effective - np.abs(extension_factor)
        x_max = cen + sigma_effective + np.abs(extension_factor)
        return x_min, x_max

    def fit_gaussians(self, x_full, y_full, ind_peak, trace, peaks, ax):
        """
        Fits single or multi-Gaussian models to the provided data to determine the best-fit parameters for the peaks of interest.

        Parameters
        ----------
        x_full : numpy.ndarray or pandas.Series
            Array of full x-values (e.g., retention times) corresponding to the data points.
        y_full : numpy.ndarray or pandas.Series
            Array of full y-values (e.g., signal intensities) corresponding to the x-values.
        ind_peak : int
            Index of the peak of interest in the data.
        trace : str
            Identifier for the trace being analyzed (e.g., which sample or dataset the peak belongs to).
        peaks : list of int
            List of indices representing the detected peaks in the data.
        ax : matplotlib.axes.Axes
            The axis object used for plotting the Gaussian fits.

        Returns
        -------
        best_x : numpy.ndarray
            Array of x-values corresponding to the best Gaussian fit (single or multi-Gaussian) for the peak of interest.
        best_fit_y : numpy.ndarray
            Array of y-values corresponding to the best Gaussian fit (single or multi-Gaussian) for the peak of interest.
        area_smooth : float
            The area under the curve for the best-fit Gaussian model, calculated using Simpson's rule.

        Notes
        -----
        - The function iteratively fits multi-Gaussian models to detect overlapping peaks and determine the best fit.
        - If a multi-Gaussian model does not provide a satisfactory fit, the function tries to fit a single Gaussian with exponential decay.
        - Gaussian parameters such as amplitude, center, and width are estimated using initial guesses and bounded constraints.
        - The function calculates boundaries for peak fitting based on the first derivative and extends the Gaussian tails beyond the peak region.
        - The best fit is determined based on the lowest root mean square error (RMSE) between the fitted Gaussian and the observed data.
        - The function returns the best-fit x and y values, along with the area under the curve using Simpson's rule for numerical integration.
        """
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
                    best_fit_params_error = pcov
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
                    single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: self.gaussian_decay(x, amp, cen, wid, dec), x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=self.gi)
                single_fitted_y = self.gaussian_decay(x, *single_popt)
                error = np.sqrt(((single_fitted_y - y) ** 2).mean())  # RMSE
                if error < best_error:
                    multi_gauss_flag = False
                    best_error = error
                    best_fit_params = single_popt
                    best_fit_params_error = single_pcov
                    best_fit_y = single_fitted_y
                    best_x = x
            except RuntimeError:
                print(f"Warning: Optimal parameters could not be found with {self.gi} iterations. Increasing iterations by a factor of 100. Please be patient.")

                # Increase maxfev by a factor of 10 and retry
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: self.gaussian_decay(x, amp, cen, wid, dec), x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=self.gi* 1000) # comment out to speed up debug
                    single_fitted_y = self.gaussian_decay(x, *single_popt)
                    error = np.sqrt(((single_fitted_y - y) ** 2).mean())  # RMSE)
                    if error < best_error:
                        multi_gauss_flag = False
                        best_error = error
                        best_fit_params = single_popt
                        best_fit_params_error = single_pcov
                        best_fit_y = single_fitted_y
                        best_x = x
                except RuntimeError:
                    print("Error: Optimal parameters could not be found even after increasing the iterations.")
        if multi_gauss_flag == True:
            # print("picked multi", trace)
            # Determine the index of the peak of interest in the multi-Gaussian fit
            amp, cen, wid = best_fit_params[best_idx_interest * 3], best_fit_params[best_idx_interest * 3 + 1], best_fit_params[best_idx_interest * 3 + 2]
            best_fit_y = self.individual_gaussian(best_x, amp, cen, wid)
            best_x, best_fit_y = self.extrapolate_gaussian(best_x, amp, cen, wid, None, best_x.min() - 1, best_x.max() + 1, step=0.01)
            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            area_smooth, area_ensemble = self.peak_area_distribution(best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ax, ind_peak, multi=True)

        else:
            # print("picked single", trace)
            x_min, x_max = self.calculate_gaus_extension_limits(best_fit_params[1], best_fit_params[2], best_fit_params[3], factor=3)
            best_x, best_fit_y = self.extrapolate_gaussian(best_x, best_fit_params[0], best_fit_params[1], best_fit_params[2], best_fit_params[3], x_min, x_max, step=0.01)
            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            area_smooth, area_ensemble = self.peak_area_distribution(best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ax, ind_peak, multi = False)

        return best_x, best_fit_y, area_smooth, area_ensemble

    def peak_area_distribution(self, params, params_uncertainty, ind, x, x_full, ax, ind_peak, multi, n_samples= 250):
        area_ensemble = []
        if multi:
            amp_i, cen_i, wid_i = params[ind * 3], params[ind * 3 + 1], params[ind * 3 + 2]
            start = 3*ind
            end = start+3
            pcov = params_uncertainty[start:end, start:end]
            # amp_unc_i, cen_unc_i, wid_unc_i = params_uncertainty[0], params_uncertainty[1], params_uncertainty[2]
            samples = np.random.multivariate_normal(np.array([amp_i, cen_i, wid_i]), pcov, size=n_samples)
            for i in range(0,n_samples):
                amp, cen, wid = samples[i,0], samples[i,1], samples[i,2]
                best_fit_y = self.individual_gaussian(x, amp, cen, wid)
                best_x, best_fit_y = self.extrapolate_gaussian(x, amp, cen, wid, None, x.min() - 1, x.max() + 1, step=0.01)
                new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
                left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
                best_x = best_x[left_boundary - 1 : right_boundary + 1]
                best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
                area_ensemble.append(simpson(y=best_fit_y, x=best_x))
            return np.mean(area_ensemble), area_ensemble
        else:
            samples = np.random.multivariate_normal(params, params_uncertainty, size=n_samples)
            for i in range(0,n_samples):
                x_min, x_max = self.calculate_gaus_extension_limits(samples[i,1], samples[i,2], samples[i,3], factor=3)
                best_x, best_fit_y = self.extrapolate_gaussian(x, samples[i,0], samples[i,1], samples[i,2], samples[i,3], x_min, x_max, step=0.01)
                new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
                left_boundary, right_boundary = self.calculate_boundaries(best_x, best_fit_y, new_ind_peak)
                best_x = best_x[left_boundary - 1 : right_boundary + 1]
                best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
                area_ensemble.append(simpson(y=best_fit_y, x=best_x))
            return np.mean(area_ensemble), area_ensemble


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
            x_fit, y_fit_smooth, area_smooth, area_ensemble = self.fit_gaussians(xdata, y_bcorr, peak_idx, trace, peak_neighborhood, ax)
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

        # Mark that we've added controls
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
        line, = ax.plot(x_values, y_filtered, "k")
        if not hasattr(self, "line_objects"):
            self.line_objects = {}
        self.line_objects[trace_idx] = line

        # Set the current x-limits based on the current window_bounds
        ax.set_xlim(self.window_bounds)

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
            peaks_total, properties = find_peaks(y_filtered, height=(min_peak_amp, self.max_peak_amp), width=0.05, prominence=self.pk_pr)
        else:
            peaks_total, properties = find_peaks(y_filtered, height=min_peak_amp, width=0.05, prominence=self.pk_pr)
        self.peaks[trace] = peaks_total
        self.peak_properties[trace] = properties
        self.peaks_indices[trace_idx] = peaks_total

    def baseline(self, x, y, deg=5, max_it=1000, tol=1e-4):
        """
        Performs baseline correction on the input signal using an iterative polynomial fitting approach.

        Parameters
        ----------
        y : numpy.ndarray or pandas.Series
            The input signal (e.g., chromatographic data) that requires baseline correction.
        deg : int, optional
            The degree of the polynomial used for fitting the baseline. Default is 5.
        max_it : int, optional
            The maximum number of iterations for the baseline fitting process. Default is 50.
        tol : float, optional
            The tolerance for stopping the iteration when the change in coefficients becomes small. Default is 1e-4.

        Returns
        -------
        base : numpy.ndarray
            The estimated baseline for the input signal.

        Notes
        -----
        - The function iteratively fits a polynomial baseline to the input signal, adjusting the coefficients until convergence
          based on the specified tolerance (`tol`).
        - If the difference between the old and new coefficients becomes smaller than the tolerance, the iteration stops early.
        - Negative values in the baseline-corrected signal are set to zero to avoid unrealistic baseline values.
        """
        original_y = y.copy()
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

        # Calculate maximum peak amplitude (3 x baseline amplitude)
        baseline_fitter = Baseline(x)
        fit, params_mask = baseline_fitter.std_distribution(y, 45)#, smooth_half_window=10)
        mask = params_mask['mask'] #  Mask for regions of signal without peaks
        # min_peak_amp = (y[mask].max()-y[mask].min())*3
        min_peak_amp = (np.std(y[mask]))*2*3 # 2 sigma times 3
        # min_peak_amp = (base.max()-base.min())*3
        # min_peak_amp = np.std(original_y-base)*3
        return base, min_peak_amp # return base


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
            self.integrated_peaks[peak_key] = {"area": 0, "rt": event.xdata, "text": text, "line": [line], "trace": trace, "area_ensemble": 0}
            self.action_stack.append(("add_line", ax, no_peak_key))
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
                                        break
                                if not peak_found:
                                    peak_key = (ax_idx, None)
                                    line = ax.axvline(ref_peak, color="red", linestyle="--", alpha=0.5)
                                    text = ax.text(ref_peak + 2, ax.get_ylim()[1] * 0.5, "No peak\n" + str(np.round(ref_peak)), color="grey", fontsize=8)
                                    no_peak_key = peak_key
                                    self.no_peak_lines[no_peak_key] = (line, text)
                                    self.integrated_peaks[peak_key] = {"area": 0, "rt": ref_peak, "trace": trace, "area_ensemble": 0}
                                    self.action_stack.append(("add_line", ax, no_peak_key))
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

    def undo_last_action(self):
        """
        Undoes the last action performed during peak selection, removing the corresponding graphical objects from the plot.

        Returns
        -------
        None
            This function updates the plot and internal data structures by undoing the last peak-related action.

        Notes
        -----
        - The function checks the `self.action_stack` for the most recent action and removes the corresponding graphical objects
          (lines, fills, text annotations) from the plot.
        - If a valid peak is found in `self.integrated_peaks`, its graphical components (line, fill, text) are removed.
        - If no graphical components are found for the given key, a message is printed.
        - If no actions are available to undo, a message indicating "No actions to undo" is printed.
        - The plot is redrawn using `plt.draw()` after the graphical components are removed.
        """
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
            # Sort peaks by retention time
            # print(matching_peaks)
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
