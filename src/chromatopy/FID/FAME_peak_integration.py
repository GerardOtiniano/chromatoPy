import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import math
from scipy.integrate import simpson
from pybaselines import Baseline

def peak_area_distribution( params, params_uncertainty, ind, x, x_full, ind_peak, multi, n_samples= 250):
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
            best_fit_y = individual_gaussian(x, amp, cen, wid)
            best_x, best_fit_y = extrapolate_gaussian(x, amp, cen, wid, None, x.min() - 1, x.max() + 1, step=0.01)
            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            area_ensemble.append(simpson(y=best_fit_y, x=best_x))
        return np.mean(area_ensemble), area_ensemble
    else:
        samples = np.random.multivariate_normal(params, params_uncertainty, size=n_samples)
        for i in range(0,n_samples):
            x_min, x_max = calculate_gaus_extension_limits(samples[i,1], samples[i,2], samples[i,3], factor=3)
            best_x, best_fit_y = extrapolate_gaussian(x, samples[i,0], samples[i,1], samples[i,2], samples[i,3], x_min, x_max, step=0.01)
            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            area_ensemble.append(simpson(y=best_fit_y, x=best_x))
        return np.mean(area_ensemble), area_ensemble


def handle_peak_selection( ax, ax_idx, xdata, y_bcorr, peak_idx, peaks):
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

    Returns
    -------
    None
        This function updates the plot and internal data structures but does not return any values.

    Notes
    -----
    - The function identifies valleys around the selected peak and determines the neighborhood boundaries for peak fitting.
    - A Gaussian fit is applied to the selected peak and its neighborhood.
    - The area under the Gaussian curve is calculated and displayed on the plot along with retention time.
    - The peak integration results are stored in `integrated_peaks` and `peak_results` for later analysis.
    - If the peak selection or fitting process encounters a runtime error, the exception is handled and ignored.
    """
    try:
        valleys = find_valleys(y_bcorr, peaks)
        # A, B, peak_neighborhood = find_peak_neighborhood_boundaries(xdata, y_bcorr, peaks[trace], valleys, peak_idx, ax, max_peaks_for_neighborhood)
        x_fit, y_fit_smooth, area_smooth, area_ensemble = fit_gaussians(xdata, y_bcorr, peak_idx, peak_neighborhood, ax)
        fill = ax.fill_between(x_fit, 0, y_fit_smooth, color="grey", alpha=0.5)
        rt_of_peak = xdata[peak_idx]
        area_text = f"Area: {area_smooth:.0f}\nRT: {rt_of_peak:.0f}"
        text_annotation = ax.annotate(area_text, xy=(rt_of_peak + 1.5, y_fit_smooth.max() * 0.5), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="grey")
        integrated_peaks[(ax_idx, peak_idx)] = {"fill": fill, "area": area_smooth, "rt": rt_of_peak, "text": text_annotation,'area_ensemble': area_ensemble}
        plt.draw()
        # if trace not in peak_results:
        #     peak_results[trace] = {"rts": [], "areas": [], "area_ensemble": []}
        # peak_results[trace]["rts"].append(rt_of_peak)
        # peak_results[trace]["areas"].append(area_smooth)  # Calculate area if needed
        # peak_results[trace]["area_ensemble"].append(area_ensemble)
    except RuntimeError:
        pass


def baseline( x, y, deg=500, max_it=1000, tol=1e-4):
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
    min_peak_amp = (np.std(y[mask]))*2*3 # 2 sigma times 3
    return base, min_peak_amp # return base

def auto_integrate_reference_peaks(df, reference_peaks, threshold=0.2):
    time = df['time'].values
    signal = df['signal'].values
    signal_smooth = smoother(signal)
    baseline, min_peak_amp = baseline(time, signal_smooth)
    y_bcorr = signal_smooth - baseline

    peak_indices, _ = find_peaks(y_bcorr, height=min_peak_amp)
    results = {}

    # fig, ax = plt.subplots(figsize=(10, 4))
    # ax.plot(time, signal, alpha=0.3, label="Raw Signal")
    # ax.plot(time, y_bcorr, label="Baseline Corrected")

    for compound, rt_list in reference_peaks.items():
        for rt in rt_list:
            peak_dists = np.abs(time[peak_indices] - rt)
            if np.any(peak_dists < threshold):
                closest_idx = peak_indices[np.argmin(peak_dists)]
                window_mask = (time >= time[closest_idx] - 3) & (time <= time[closest_idx] + 3)
                x = time[window_mask]
                y = y_bcorr[window_mask]

                amp = y.max()
                cen = x[np.argmax(y)]
                wid = 0.5
                try:
                    popt, _ = curve_fit(individual_gaussian, x, y, p0=[amp, cen, wid])
                    y_fit = individual_gaussian(x, *popt)
                    area = simpson(y_fit, x)
                    # ax.fill_between(x, 0, y_fit, alpha=0.3, label=f\"{compound}: {area:.1f}\")
                    results.setdefault(compound, []).append({'rt': rt, 'area': area})
                except RuntimeError:
                    results.setdefault(compound, []).append({'rt': rt, 'area': np.nan})
            else:
                # ax.axvline(rt, color='red', linestyle='--', alpha=0.5)
                # ax.text(rt + 1, y_bcorr.max() * 0.5, f\"{compound}\nNo Peak\", fontsize=8, color='gray')
                results.setdefault(compound, []).append({'rt': rt, 'area': 0})

    # ax.legend()
    # ax.set_title(\"Auto Peak Integration by Compound\")
    # ax.set_xlabel(\"Time (min)\")
    # ax.set_ylabel(\"Signal\") 
    plt.tight_layout()
    plt.show()
    return results

# %%
import pandas as pd
# peak_timing = [6.31, 6.475, 6.545, 6.575, 6.613333, 6.671667, 6.868333, 6.935, 7.018333, 7.386667, 7.676667, 7.805, 8.28, 8.823333, 9.431667, 10.113333, 10.568333, 10.836667, 11.633333,12.435, 13.296667, 13.698333, 13.701667, 13.906667, 14.128333, 14.803333, 14.81, 14.995, 15.851667, 15.86, 15.863333, 16.581667, 16.585, 16.723333, 17.58, 17.838333, 18.415, 18.418333, 18.968333, 19.23, 19.241667, 19.92, 19.926667, 20.055]
peak_timing = [6.31,14.995, 13.3]

# 1. Import data
df = pd.read_csv('/Users/gerard/Desktop/TG054.csv')
df['Value (pA)'] = pd.to_numeric(df['Value (pA)'], errors='coerce')
df = df[df['Time (min)']>2]
df.reset_index(inplace=True)
xdata = df['Time (min)']
ydata = df['Value (pA)']


# 2. Apply and correct for baseline 
base, min_peak_amp = baseline(xdata, ydata, deg=5, max_it=1000, tol=1e-4)
y_bcorr = ydata-base
# 3. Identify peaks
peak_indices, peak_properties = find_peaks(y_bcorr, height=min_peak_amp, prominence=0.01)

# 4. Identify peak presence
matched_indices, presence_flags = zip(*[
    (peak_indices[np.argmin(np.abs(xdata.iloc[peak_indices] - pt))], True)
    if np.min(np.abs(xdata.iloc[peak_indices] - pt)) <= 5/60
    else (None, False)
    for pt in peak_timing
])
matched_indices = list(matched_indices)

fig = plt.figure()
plt.plot(xdata, y_bcorr, c= 'k', linewidth=2)
# plt.plot(df['Time (min)'], y_bcorr, c= 'red', linewidth=2)
plt.scatter(df.loc[matched_indices, 'Time (min)'],y_bcorr[matched_indices], c= 'red', ec='k', alpha=0.5)
plt.show()

# %%
import warnings 

gi = 1000
pk_sns = 0.01
peak_idx = 7796
max_peaks_for_neighborhood = 5
smoothing_params=[12, 3]

fig = plt.figure()
plt.plot(xdata, y_bcorr, c= 'k', linewidth=0.5, linestyle='--')
# plt.plot(df['Time (min)'], y_bcorr, c= 'red', linewidth=2)
plt.scatter(df.loc[matched_indices, 'Time (min)'],y_bcorr[matched_indices], c= 'red', ec='k', alpha=0.5)
# PEAK ID
valleys = find_valleys(y_bcorr, peak_indices)
for peak_idx in matched_indices:
    A, B, peak_neighborhood = find_peak_neighborhood_boundaries(
        x=xdata, y_smooth=y_bcorr, peaks = peak_indices, valleys=valleys, peak_idx=peak_idx, max_peaks=max_peaks_for_neighborhood)
    x_fit, y_fit_smooth, area_smooth, area_ensemble = fit_gaussians(xdata, y_bcorr, peak_idx, peak_neighborhood)
    fill = plt.fill_between(x_fit, 0, y_fit_smooth, color="grey", alpha=0.5)
plt.xlim(5,20)
plt.show()
# rt_of_peak = xdata[peak_idx]
# area_text = f"Area: {area_smooth:.0f}\nRT: {rt_of_peak:.0f}"
# text_annotation = ax.annotate(area_text, xy=(rt_of_peak + 1.5, y_fit_smooth.max() * 0.5), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="grey")
# integrated_peaks[(ax_idx, peak_idx)] = {"fill": fill, "area": area_smooth, "rt": rt_of_peak, "text": text_annotation, 'area_ensemble': area_ensemble}
# plt.draw()
# peak_results[trace] = {"rts": [], "areas": [], "area_ensemble": []}
# self.peak_results[trace]["rts"].append(rt_of_peak)
# self.peak_results[trace]["areas"].append(area_smooth)  # Calculate area if needed
# self.peak_results[trace]["area_ensemble"].append(area_ensemble)
# %%
def smoother(y, param_0 = None, param_1 = None):
    if param_0 == None:
        param_0 = smoothing_params[0]
    if param_1 == None:
        param_1 = smoothing_params[1]
    # return savgol_filter(y, self.smoothing_params[0], self.smoothing_params[1], deriv=0, mode='interp')
    return savgol_filter(y, param_0, param_1)

def find_valleys(y, peaks, peak_oi=None):
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

def find_peak_neighborhood_boundaries(x, y_smooth, peaks, valleys, peak_idx, max_peaks):
    peak_distances = np.abs(x[peaks] - x[peak_idx])
    closest_peaks_indices = np.argsort(peak_distances)[:max_peaks]
    closest_peaks = np.sort(peaks[closest_peaks_indices])

    overlapping_peaks = []
    extended_boundaries = {}
    # Analyze each of the closest peaks
    for peak in closest_peaks:
        peak_pos = np.where(peak == peaks)
        l_lim = peak_properties["left_bases"][peak_pos][0]
        r_lim = peak_properties["right_bases"][peak_pos][0]
        heights, means, stddevs = estimate_initial_gaussian_params(x[l_lim : r_lim + 1], y_smooth[l_lim : r_lim + 1], peak)
        height, mean, stddev = heights[0], means[0], stddevs[0]

        # Fit Gaussian and get best fit parameters
        try:
            popt, _ = curve_fit(individual_gaussian, x, y_smooth, p0=[height, mean, stddev], maxfev=gi)
        except RuntimeError:
            popt, _ = curve_fit(individual_gaussian, x, y_smooth, p0=[height, mean, stddev], maxfev=gi*100)
        # popt, _ = curve_fit(gaussian, x, y_smooth, p0=[height, mean, stddev, 0.1], maxfev=gi)
        # Extend Gaussian fit limits
        x_min, x_max = calculate_gaus_extension_limits(popt[1], popt[2], 0, factor=3)
        extended_x, extended_y = extrapolate_gaussian(x, popt[0], popt[1], popt[2], None, x_min - 2, x_max + 2)
        # Find the boundaries based on the derivative test
        peak_x_value = x[peak]
        n_peak_idx = np.argmin(np.abs(extended_x - peak_x_value))
        left_idx, right_idx = calculate_boundaries(extended_x, extended_y, n_peak_idx)
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




# Gaussian fitting
def calculate_gaus_extension_limits(cen, wid, decay, factor=3):  # decay, factor=3):
    sigma_effective = wid * factor  # Adjust factor for tail thinness
    extension_factor = 1 / decay if decay != 0 else sigma_effective  # Use decay to modify the extension if applicable
    x_min = cen - sigma_effective - np.abs(extension_factor)
    x_max = cen + sigma_effective + np.abs(extension_factor)
    return x_min, x_max

def extrapolate_gaussian(x, amp, cen, wid, decay, x_min, x_max, step=0.01):
    extended_x = np.arange(x_min, x_max, step)
    if decay is None:
        extended_y = individual_gaussian(extended_x, amp, cen, wid)
    else:
        extended_y = gaussian_decay(extended_x, amp, cen, wid, decay)
    return extended_x, extended_y
def calculate_boundaries( x, y, ind_peak):
    smooth_y = smoother(y)
    velocity, X1 = forward_derivative(x, smooth_y)
    velocity /= np.max(np.abs(velocity))
    smooth_velo = smoother(velocity)
    dt = int(np.ceil(0.025 / np.mean(np.diff(x))))
    A = np.where(smooth_velo[: ind_peak - 3 * dt] < pk_sns)[0]  # 0.05)[0]
    B = np.where(smooth_velo[ind_peak + 3 * dt :] > -pk_sns)[0]  # -0.05)[0]
    if A.size > 0:
        A = A[-1] + 1
    else:
        A = 1
    if B.size > 0:
        B = B[0] + ind_peak + 3 * dt - 1
    else:
        B = len(x) - 1
    return A, B

def fit_gaussians(x_full, y_full, ind_peak, peaks):
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
        left_boundary, _ = calculate_boundaries(x_full, y_full, np.min(current_peaks))
        _, right_boundary = calculate_boundaries(x_full, y_full, np.max(current_peaks))
        x = x_full[left_boundary : right_boundary + 1]
        y = y_full[left_boundary : right_boundary + 1]
        index_of_interest = np.where(current_peaks == ind_peak)[0][0]
        initial_guesses = []
        bounds_lower = []
        bounds_upper = []
        for peak in current_peaks:
            height, center, width = estimate_initial_gaussian_params(x, y, peak)  # peak)
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
                popt, pcov = curve_fit(multigaussian, x, y, p0=initial_guesses, method="dogbox", bounds=bounds, maxfev=gi)  # , ftol=1e-4, xtol=1e-4)
            fitted_y = multigaussian(x, *popt)
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
        left_boundary, right_boundary = calculate_boundaries(x_full, y_full, ind_peak)
        x = x_full[left_boundary : right_boundary + 1]
        y = y_full[left_boundary : right_boundary + 1]
        height, center, width = estimate_initial_gaussian_params(x, y, ind_peak)
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
                single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: gaussian_decay(x, amp, cen, wid, dec), x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=gi)
            single_fitted_y = gaussian_decay(x, *single_popt)
            error = np.sqrt(((single_fitted_y - y) ** 2).mean())  # RMSE
            if error < best_error:
                multi_gauss_flag = False
                best_error = error
                best_fit_params = single_popt
                best_fit_params_error = single_pcov
                best_fit_y = single_fitted_y
                best_x = x
        except RuntimeError:
            print(f"Warning: Optimal parameters could not be found with {gi} iterations. Increasing iterations by a factor of 100. Please be patient.")

            # Increase maxfev by a factor of 10 and retry
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    single_popt, single_pcov = curve_fit(lambda x, amp, cen, wid, dec: gaussian_decay(x, amp, cen, wid, dec), x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=gi* 1000) # comment out to speed up debug
                single_fitted_y = gaussian_decay(x, *single_popt)
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
        # Determine the index of the peak of interest in the multi-Gaussian fit
        amp, cen, wid = best_fit_params[best_idx_interest * 3], best_fit_params[best_idx_interest * 3 + 1], best_fit_params[best_idx_interest * 3 + 2]
        best_fit_y = individual_gaussian(best_x, amp, cen, wid)
        best_x, best_fit_y = extrapolate_gaussian(best_x, amp, cen, wid, None, best_x.min() - 1, best_x.max() + 1, step=0.01)
        new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
        left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak)
        best_x = best_x[left_boundary - 1 : right_boundary + 1]
        best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
        area_smooth, area_ensemble = peak_area_distribution(best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ind_peak, multi=True)

    else:
        x_min, x_max = calculate_gaus_extension_limits(best_fit_params[1], best_fit_params[2], best_fit_params[3], factor=3)
        best_x, best_fit_y = extrapolate_gaussian(best_x, best_fit_params[0], best_fit_params[1], best_fit_params[2], best_fit_params[3], x_min, x_max, step=0.01)
        new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
        left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak)
        best_x = best_x[left_boundary - 1 : right_boundary + 1]
        best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
        area_smooth, area_ensemble = peak_area_distribution(best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ind_peak, multi = False)
    return best_x, best_fit_y, area_smooth, area_ensemble

def peak_area_distribution( params, params_uncertainty, ind, x, x_full, ind_peak, multi, n_samples= 250):
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
            best_fit_y = individual_gaussian(x, amp, cen, wid)
            best_x, best_fit_y = extrapolate_gaussian(x, amp, cen, wid, None, x.min() - 1, x.max() + 1, step=0.01)
            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            area_ensemble.append(simpson(y=best_fit_y, x=best_x))
        return np.mean(area_ensemble), area_ensemble
    else:
        samples = np.random.multivariate_normal(params, params_uncertainty, size=n_samples)
        for i in range(0,n_samples):
            x_min, x_max = calculate_gaus_extension_limits(samples[i,1], samples[i,2], samples[i,3], factor=3)
            best_x, best_fit_y = extrapolate_gaussian(x, samples[i,0], samples[i,1], samples[i,2], samples[i,3], x_min, x_max, step=0.01)
            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            area_ensemble.append(simpson(y=best_fit_y, x=best_x))
        return np.mean(area_ensemble), area_ensemble
    
def individual_gaussian( x, amp, cen, wid):
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))

def estimate_initial_gaussian_params( x, y, peak):
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

def multigaussian( x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        cen = params[i + 1]
        wid = params[i + 2]
        y += amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))
    return y

def gaussian_decay( x, amp, cen, wid, dec):
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2)) * np.exp(-dec * abs(x - cen))
def forward_derivative(x, y):
    fd = np.diff(y) / np.diff(x)
    x_n = x[:-1]
    return fd, x_n

class FIDAnalyzer:
    def __init__(self, df, window_bounds, gaus_iterations, sample_name, is_reference, max_peaks, sw, sf, pk_sns, pk_pr, max_PA, reference_peaks=None):
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
    
    