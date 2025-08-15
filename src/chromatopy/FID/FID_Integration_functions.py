import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import math
from scipy.integrate import simpson
from pybaselines import Baseline
import warnings 
import pandas as pd
from tqdm import tqdm
from scipy.special import erf
import matplotlib.pyplot as plt

# Functions
def baseline( x, y, deg=5, max_it=1000, tol=1e-4):
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

# def smoother(y, param_0, param_1, mode = "interp"):# "constant"):
#     return savgol_filter(y, param_0, param_1, mode=mode)
def smoother(y, window_length, polyorder):
    from scipy.signal import savgol_filter

    if len(y) < 3:
        return y  # don't try to smooth tiny series

    # Adjust window_length to be <= len(y) and an odd integer
    window_length = min(window_length, len(y) - 1 if len(y) % 2 == 0 else len(y))
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, polyorder + 2 + (polyorder % 2))  # ensure still valid

    return savgol_filter(y, window_length=window_length, polyorder=polyorder)

def find_peak_neighborhood_boundaries(x, y_smooth, peaks, valleys, peak_idx, max_peaks, peak_properties, gi, smoothing_params, pk_sns):
    overlapping_peaks = []
    extended_boundaries = {}
    # Analyze each of the closest peaks
    for peak in peaks: #closest_peaks:
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
        # Extend Gaussian fit limits
        x_min, x_max = calculate_gaus_extension_limits(popt[1], popt[2], 0, factor=3)
        extended_x, extended_y = extrapolate_gaussian(x, popt[0], popt[1], popt[2], None, x_min - 2, x_max + 2)
        # Find the boundaries based on the derivative test
        peak_x_value = x[peak]
        n_peak_idx = np.argmin(np.abs(extended_x - peak_x_value))
        left_idx, right_idx = calculate_boundaries(extended_x, extended_y, n_peak_idx, smoothing_params, pk_sns)
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
def calculate_gaus_extension_limits(cen, wid, decay, factor=2, max_tail_sigma=2):#5):
    sigma_effective = wid * factor  # Adjust factor for tail thinness
    if decay <= 0:
        tail = sigma_effective * max_tail_sigma
    else:
        tail = min(1/decay, sigma_effective * max_tail_sigma)
    return cen - sigma_effective-tail, cen+sigma_effective+tail

def extrapolate_gaussian(x, amp, cen, wid, skew=None, x_min=None, x_max=None, step=0.0001):
    if x_min is None: x_min = cen - 3 * wid
    if x_max is None: x_max = cen + 3 * wid
    extended_x = np.arange(x_min, x_max, step)
    if skew is None:
        extended_y = individual_gaussian(extended_x, amp, cen, wid)
    else:
        extended_y = skewed_gaussian(extended_x, amp, cen, wid, skew)
    return extended_x, extended_y

def extrapolate_gaussian_decay(amp, cen, wid, dec, x_min=None, x_max=None, step=1e-4):
    if x_min is None:
        x_min = cen - 3 * wid
    if x_max is None:
        x_max = cen + 3 * wid
    xs = np.arange(x_min, x_max, step)
    ys = gaussian_decay(xs, amp, cen, wid, dec)
    return xs, ys

def calculate_boundaries(x, y, ind_peak, smoothing_params, pk_sns):
    smooth_y = smoother(y, smoothing_params[0], smoothing_params[1])
    velocity, X1 = forward_derivative(x, smooth_y)
    velocity /= np.max(np.abs(velocity))
    if smoothing_params[0] > len(velocity):
        smoother_val = len(velocity)-1
    else: smoother_val = smoothing_params[0]
    smooth_velo = smoother(velocity, smoother_val, smoothing_params[1])
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


def calculate_boundaries_acceleration(x, y, ind_peak, smoothing_params, pk_sns):
    smooth_y = smoother(y, smoothing_params[0], smoothing_params[1])
    velocity, _ = forward_derivative(x, smooth_y)
    acceleration, _ = forward_derivative(x[:-1], velocity)
    acceleration /= np.max(np.abs(acceleration))
    smoother_val = min(smoothing_params[0], len(acceleration) - 1)
    smooth_accel = smoother(acceleration, smoother_val, smoothing_params[1])
    left_zone = smooth_accel[:ind_peak]
    right_zone = smooth_accel[ind_peak:]
    if len(left_zone) > 0:
        A = np.argmax(left_zone)
    else:
        A = 1
    if len(right_zone) > 0:
        B = np.argmax(right_zone) + ind_peak
    else:
        B = len(x) - 1
    return A, B
# def calculate_boundaries_acceleration(x, y, ind_peak, smoothing_params, pk_sns):
#     smooth_y = smoother(y, smoothing_params[0], smoothing_params[1])

#     # 1) Derivatives
#     vel, _ = forward_derivative(x, smooth_y)               # len = n-1
#     if vel.size == 0:
#         return 0, max(len(x) - 1, 0)

#     acc, _ = forward_derivative(x[:-1], vel)               # len = n-2
#     if acc.size == 0:
#         return 0, max(len(x) - 1, 0)

#     # 2) Normalize safely (avoid div-by-zero)
#     amax = np.max(np.abs(acc))
#     if amax > 0:
#         acc = acc / amax

#     # 3) Smoothing window must be valid (odd & >= polyorder+2)
#     win = min(smoothing_params[0], len(acc) - 1)
#     if win % 2 == 0:
#         win -= 1
#     win = max(win, smoothing_params[1] + 2 + (smoothing_params[1] % 2))
#     win = min(win, max(len(acc) - 1, 1))
#     if win < 3:  # too short to smooth; skip smoothing
#         smooth_acc = acc
#     else:
#         smooth_acc = smoother(acc, win, smoothing_params[1])

#     # 4) Split around the peak index (ind_peak refers to x/y index)
#     left_zone  = smooth_acc[:max(ind_peak, 0)]
#     right_zone = smooth_acc[max(ind_peak, 0):]

#     # 5) Pick extremum on each side; clamp to valid [0, len(x)-1]
#     A = int(np.argmax(left_zone)) if left_zone.size > 0 else 0
#     B = int(np.argmax(right_zone)) + max(ind_peak, 0) if right_zone.size > 0 else len(x) - 1

#     # 6) Ensure A < B and avoid A-1 underflow in downstream slicing
#     A = max(A, 1)                  # so slice [A-1: ...] doesn’t go negative
#     B = min(B, len(x) - 1)

#     if B <= A:
#         # conservative fallback: symmetric window around the peak
#         pad = max(int(0.02 / max(np.mean(np.diff(x)), 1e-9)), 10)  # ~window in samples
#         A = max(ind_peak - pad, 1)
#         B = min(ind_peak + pad, len(x) - 1)

#     return A, B

def fit_gaussians(x_full, y_full, ind_peak, peaks, smoothing_params, pk_sns, gi, mode="both"):
    if mode not in {"single", "multi", "both"}:
        raise ValueError("mode must be 'single', 'multi', or 'both'")
    # figy = plt.figure()
    results = []
    
    # --- MULTI-GAUSSIAN ---
    if mode in {"multi", "both"}:
        result = _fit_multi_gaussian(x_full, y_full, ind_peak, peaks, smoothing_params, pk_sns, gi)
        if result is not None:
            best_x, best_fit_y, best_fit_params, best_fit_params_error, best_error, best_idx_interest = result
            results.append({
                "name": "multi",
                "x": best_x,
                "y": best_fit_y,
                "params": best_fit_params,
                "pcov": best_fit_params_error,
                "error": best_error,
                "idx_interest": best_idx_interest,
                "multi_flag": True})

    # --- SINGLE-GAUSSIAN ---
    if mode in {"single", "both"}:
        print("debug 1.1")
        result = _fit_single_gaussian(x_full, y_full, ind_peak, smoothing_params, pk_sns, gi, current_best_error=float("inf"))
        print("debug 1.2")
        if result is not None:
            best_x, best_fit_y, best_fit_params, best_fit_params_error, best_error = result
            results.append({
                "name": "single",
                "x": best_x,
                "y": best_fit_y,
                "params": best_fit_params,
                "pcov": best_fit_params_error,
                "error": best_error,
                "multi_flag": False,
                "idx_interest": None})

    # --- ASYMMETRIC MODEL (always run) ---
    
    print("debug 1.3")
    result = _fit_asymmetric_gaussian(x_full, y_full, ind_peak, smoothing_params, pk_sns, gi, current_best_error=float("inf"))
    print("debug 1.4")
    if result is not None:
        best_x, best_fit_y, best_fit_params, best_fit_params_error, best_error = result
        results.append({
            "name": "asymmetric",
            "x": best_x,
            "y": best_fit_y,
            "params": best_fit_params,
            "pcov": best_fit_params_error,
            "error": best_error,
            "multi_flag": False,
            "idx_interest": None})
    if not results:
        raise RuntimeError(f"No valid fit found for peak at index {ind_peak}")

    best_result = min(results, key=lambda r: r["error"])
    print("debug 1.5")
    # --- Process best fit output ---
    best_x = best_result["x"]
    best_fit_y = best_result["y"]
    best_fit_params = best_result["params"]
    best_fit_params_error = best_result["pcov"]
    best_idx_interest = best_result.get("idx_interest", None)
    multi_gauss_flag = best_result["multi_flag"]
    model_used = best_result["name"]
    print("debug 1.6")
    print(best_x)
    print(best_fit_y)
    # --- Extend fit + calculate area ---
    if multi_gauss_flag:
        amp, cen, wid = best_fit_params[best_idx_interest * 3: best_idx_interest * 3 + 3]
        best_fit_y = individual_gaussian(best_x, amp, cen, wid)
        best_x, best_fit_y = extrapolate_gaussian(best_x, amp, cen, wid, None, best_x.min() - 1, best_x.max() + 1, step=0.0001)
        new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
        left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak, smoothing_params, pk_sns)
        best_x = best_x[left_boundary - 1: right_boundary + 1]
        best_fit_y = best_fit_y[left_boundary - 1: right_boundary + 1]
        area_smooth, area_ensemble = peak_area_distribution(best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ind_peak, multi=True, smoothing_params=smoothing_params, pk_sns=pk_sns)
    # else:
    #     amp, cen, wid = best_fit_params[:3]
    #     tail_factor = 3
    #     x_min, x_max = calculate_gaus_extension_limits(cen, wid, 0, factor=tail_factor)
    #     # print("debug 1.6")
    #     if model_used == "asymmetric":
    #         alpha = best_fit_params[3]
    #         best_x, best_fit_y = extrapolate_gaussian(best_x, amp, cen, wid, alpha, x_min, x_max, step=0.0001)
    #         # print("debug 1.6.1")
    #     else:
    #         best_x, best_fit_y = extrapolate_gaussian(best_x, amp, cen, wid, None, x_min, x_max, step=0.0001)
    #         # print("debug 1.6.2")
    #     new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
    #     # print("debug 1.7")
    #     left_boundary, right_boundary = calculate_boundaries_acceleration(best_x, best_fit_y, new_ind_peak, smoothing_params, pk_sns)
    #     # print("debug 1.8")
    #     area_smooth, area_ensemble = peak_area_distribution(best_fit_params, best_fit_params_error, best_idx_interest, best_x, x_full, ind_peak, multi=False, smoothing_params=smoothing_params, pk_sns=pk_sns)
    #     # print("debug 1.9")
    else:
        amp, cen, wid = best_fit_params[:3]
        tail_factor = 1.5#3
        print("debug 1.8")
        if model_used == "asymmetric":
            # skewed Gaussian (alpha)
            print("1.8.1")
            alpha = best_fit_params[3]
            print("1.8.2")
            x_min, x_max = calculate_gaus_extension_limits(cen, wid, 0, factor=tail_factor)
            print("1.8.3")
            best_x, best_fit_y = extrapolate_gaussian(
                best_x, amp, cen, wid, alpha, x_min, x_max, step=0.0001)
            print("1.8.4")
            print(best_x)
            print(best_fit_y)
        elif model_used == "single":
            print("debug 1.9")
            # gaussian_decay (dec)
            dec = best_fit_params[3]
            x_min, x_max = calculate_gaus_extension_limits(cen, wid, dec, factor=tail_factor)
            best_x, best_fit_y = extrapolate_gaussian_decay(
                amp, cen, wid, dec, x_min, x_max, step=0.0001)
        else:
            print("debug 1.10")
            # pure symmetric Gaussian fallback
            x_min, x_max = calculate_gaus_extension_limits(cen, wid, 0, factor=tail_factor)
            print("debug 1.11")
            best_x, best_fit_y = extrapolate_gaussian(
                best_x, amp, cen, wid, None, x_min, x_max, step=0.0001)
        print("debug 1.12")
        new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
        left_boundary, right_boundary = calculate_boundaries_acceleration(
            best_x, best_fit_y, new_ind_peak, smoothing_params, pk_sns)
        print("debug 1.13")
        print(new_ind_peak)
        print(left_boundary)
        print(right_boundary)
        area_smooth, area_ensemble = peak_area_distribution(
            best_fit_params, best_fit_params_error, best_idx_interest,
            best_x, x_full, ind_peak, multi=False,
            smoothing_params=smoothing_params, pk_sns=pk_sns)
        print(area_smooth)
    return best_x, best_fit_y, area_smooth, area_ensemble, best_result


def _fit_multi_gaussian(x_full, y_full, ind_peak, peaks, smoothing_params, pk_sns, gi):
    current_peaks = np.sort(np.append(peaks, ind_peak))
    best_fit_y = None
    best_fit_params = None
    best_fit_params_error = None
    best_x = None
    best_error = float("inf")
    best_idx_interest = None

    while len(current_peaks) > 1:
        left, _ = calculate_boundaries(x_full, y_full, np.min(current_peaks), smoothing_params, pk_sns)
        _, right = calculate_boundaries(x_full, y_full, np.max(current_peaks), smoothing_params, pk_sns)
        x = x_full[left:right + 1]
        y = y_full[left:right + 1]
        index_of_interest = np.where(current_peaks == ind_peak)[0][0]

        p0, bounds = [], ([], [])
        for peak in current_peaks:
            h, c, w = estimate_initial_gaussian_params(x, y, peak)
            p0.extend([h[0], c[0], w[0]])
            bounds[0].extend([0.1 * y_full[peak], x_full[peak] - 0.15, max(w[0] - 0.1, 0)])
            bounds[1].extend([1 + y_full[peak], x_full[peak] + 0.15, 0.5 + w[0]])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(multigaussian, x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=gi)
            fitted_y = multigaussian(x, *popt)
            error = np.sqrt(np.mean((fitted_y - y) ** 2))
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
        if distances.size:
            current_peaks = np.delete(current_peaks, np.argmax(distances))

    if best_fit_params is not None:
        return best_x, best_fit_y, best_fit_params, best_fit_params_error, best_error, best_idx_interest
    return None

def _fit_single_gaussian(x_full, y_full, ind_peak, smoothing_params, pk_sns, gi, current_best_error):
    left, right = calculate_boundaries(x_full, y_full, ind_peak, smoothing_params, pk_sns)
    x = x_full[left:right + 1]
    y = y_full[left:right + 1]
    h, c, w = estimate_initial_gaussian_params(x, y, ind_peak)
    center_idx = (np.abs(x - c[0])).argmin()
    decay_init = estimate_initial_decay(x, y, center_idx)
    p0 = [h[0], c[0], w[0], decay_init]
    # p0 = [h[0], c[0], w[0], 0.1]
    bounds = ([0.9 * y_full[ind_peak], x_full[ind_peak] - 0.1, 0.5 * w[0], 0.01],
              [1 + y_full[ind_peak], x_full[ind_peak] + 0.1, 1.5 * w[0], 2])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(gaussian_decay, x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=gi)
        fitted_y = gaussian_decay(x, *popt)
        error = np.sqrt(np.mean((fitted_y - y) ** 2))
        if error < current_best_error:
            return x, fitted_y, popt, pcov, error
    except RuntimeError:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(gaussian_decay, x, y, p0=p0, method="dogbox", bounds=bounds, maxfev=gi * 1000)
            fitted_y = gaussian_decay(x, *popt)
            error = np.sqrt(np.mean((fitted_y - y) ** 2))
            if error < current_best_error:
                return x, fitted_y, popt, pcov, error
        except RuntimeError:
            tqdm.write("Error: Optimal parameters could not be found even after increasing iterations.")
    return None

def _fit_asymmetric_gaussian(x_full, y_full, ind_peak, smoothing_params, pk_sns, gi, current_best_error):
    left, right = calculate_boundaries(x_full, y_full, ind_peak, smoothing_params, pk_sns)
    x = x_full[left:right + 1]
    y = y_full[left:right + 1]

    h, c, w = estimate_initial_gaussian_params(x, y, ind_peak)

    # Sanitize estimates
    amp = max(h[0], 1e-5)
    cen = c[0]
    wid = max(w[0], 1e-5)
    alpha = 0.0  # Start symmetric

    p0 = [amp, cen, wid, alpha]
    bounds = (
        [1e-5, cen - 0.1, 1e-5, -10],  # lower bounds
        [10 * amp, cen + 0.1, 10 * wid, 10]  # upper bounds
    )

    try:
        popt, pcov = curve_fit(skewed_gaussian, x, y, p0=p0, bounds=bounds, maxfev=gi)
        fitted_y = skewed_gaussian(x, *popt)
        error = np.sqrt(np.mean((fitted_y - y) ** 2))
        if error < current_best_error:
            return x, fitted_y, popt, pcov, error
    except RuntimeError:
        pass
    return None

# def draw_positive_mvnorm(mu, cov, n_samples, max_attempts=10000):
#     """
#     Draw exactly n_samples from N(mu, cov) but only keep those with wid>0.
#     We no longer filter on decay here.
#     """
#     mu = np.asarray(mu)
#     out = []
#     attempts = 0

#     while len(out) < n_samples and attempts < max_attempts:
#         to_draw = n_samples - len(out)
#         batch = np.random.multivariate_normal(mu, cov, size=to_draw)
#         # only require width > 0 (batch[:,2])
#         mask = batch[:,2] > 0
#         out.extend(batch[mask].tolist())
#         attempts += 1

#     if len(out) < n_samples:
#         raise RuntimeError(
#             f"Could only draw {len(out)} valid samples after {attempts} attempts "
#             f"(needed {n_samples}).")

#     return np.array(out[:n_samples])
def draw_positive_mvnorm(mu, cov, n_samples, max_attempts=10000):
    mu = np.asarray(mu)
    out = []
    attempts = 0
    while len(out) < n_samples and attempts < max_attempts:
        to_draw = n_samples - len(out)
        batch = np.random.multivariate_normal(mu, cov, size=to_draw)
        # require amp>0 and wid>0
        mask = (batch[:,0] > 0) & (batch[:,2] > 0)
        out.extend(batch[mask].tolist())
        attempts += 1
    if len(out) < n_samples:
        raise RuntimeError(f"Could only draw {len(out)} valid samples after {attempts} attempts (needed {n_samples}).")
    return np.array(out[:n_samples])

def draw_positive_mvnorm3(mu3, cov3, n_samples, max_attempts=10000):
    """
    Sample exactly n_samples from N(mu3, cov3) with amp>0 and wid>0.
    mu3 = [amp, cen, wid]
    """
    mu3 = np.asarray(mu3)
    out = []
    attempts = 0
    while len(out) < n_samples and attempts < max_attempts:
        to_draw = n_samples - len(out)
        batch = np.random.multivariate_normal(mu3, cov3, size=to_draw)
        mask = (batch[:, 0] > 0) & (batch[:, 2] > 0)  # amp>0, wid>0
        out.extend(batch[mask].tolist())
        attempts += 1
    if len(out) < n_samples:
        raise RuntimeError(
            f"Could only draw {len(out)} valid samples after {attempts} attempts (needed {n_samples})."
        )
    return np.array(out[:n_samples])


def peak_area_distribution( params, params_uncertainty, ind, x, x_full, ind_peak, multi, smoothing_params, pk_sns, n_samples= 100):
    area_ensemble = []
    if multi:
        amp_i, cen_i, wid_i = params[ind * 3], params[ind * 3 + 1], params[ind * 3 + 2]
        start = 3*ind
        end = start+3
        pcov = params_uncertainty[start:end, start:end]
        # amp_unc_i, cen_unc_i, wid_unc_i = params_uncertainty[0], params_uncertainty[1], params_uncertainty[2]
        # samples = np.random.multivariate_normal(np.array([amp_i, cen_i, wid_i]), pcov, size=n_samples)
        # for i in range(0,n_samples):
        #     amp, cen, wid = samples[i,0], samples[i,1], samples[i,2]
        #     best_fit_y = individual_gaussian(x, amp, cen, wid)
        #     best_x, best_fit_y = extrapolate_gaussian(x, amp, cen, wid, None, x.min() - 1, x.max() + 1, step=0.0001)
        #     new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
        #     left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak, smoothing_params, pk_sns)
        #     best_x = best_x[left_boundary - 1 : right_boundary + 1]
        #     best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
        #     area_ensemble.append(simpson(y=best_fit_y, x=best_x))
        # return np.median(area_ensemble), area_ensemble
        samples = draw_positive_mvnorm3( np.array([amp_i, cen_i, wid_i]), pcov, n_samples)
        for i in range(n_samples):
            amp, cen, wid = samples[i]
            wid = max(wid, 1e-6)  # numerical safety
            # generate the curve for this component
            best_fit_y = individual_gaussian(x, amp, cen, wid)
            best_x, best_fit_y = extrapolate_gaussian(x, amp, cen, wid, None, x.min() - 1, x.max() + 1, step=1e-4)
            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak, smoothing_params, pk_sns)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            # clip tiny negatives
            best_fit_y = np.maximum(best_fit_y, 0)
            area_ensemble.append(simpson(y=best_fit_y, x=best_x))
    else:
        # samples = np.random.multivariate_normal(params, params_uncertainty, size=n_samples)
        # print("debug 1.8.1")
        # debug_param_distribution(params, params_uncertainty, n_draw=5000)
        # print("debug 1.8.1.5")
        print("debug 1.13.1")
        samples = draw_positive_mvnorm(params, params_uncertainty,n_samples)
        # # print("debug 1.8.2")
        # x = 0
        # for i in range(0,n_samples): 
        #     # print(f"1.8.2.{x}.a")
        #     amp, cen, wid, decay = samples[i]
        #     # print(f"1.8.2.{x}.b")
        #     decay_eff = np.clip(decay, 0.0, 15.0)
        #     wid   = max(abs(wid), 1e-6)
        #     decay = max(decay, 1e-6)
        #     # print(f"1.8.2.{x}.c")
        #     x_min, x_max = calculate_gaus_extension_limits(cen, wid, decay_eff, factor=3)
        #     # print(f"1.8.2.{x}.d")
        #     best_x, best_fit_y = extrapolate_gaussian(x, amp, cen, wid, decay_eff, x_min, x_max, step=1e-4)
        #     # x_min, x_max = calculate_gaus_extension_limits(cen, wid, decay, factor=3)
        #     # best_x, best_fit_y = extrapolate_gaussian(x, samples[i,0], samples[i,1], samples[i,2], samples[i,3], x_min, x_max, step=0.0001)
        #     new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
        #     # print(f"1.8.2.{x}.e")
        #     left_boundary, right_boundary = calculate_boundaries(best_x, best_fit_y, new_ind_peak, smoothing_params, pk_sns)
        #     # print(f"1.8.2.{x}.f")
        #     best_x = best_x[left_boundary - 1 : right_boundary + 1]
        #     # print(f"1.8.2.{x}.g")
        #     best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
        #     # print(f"1.8.2.{x}.h")
        #     area_ensemble.append(simpson(y=best_fit_y, x=best_x))
        #     # print(f"1.8.2.{x}.i")
        #     x=+1
        # return np.median(area_ensemble), area_ensemble
        for i in range(n_samples): 
            print("debug 1.13.2")
            amp, cen, wid, decay = samples[i]
            print("debug 1.13.3")
            print(amp)
            print(cen)
            print(wid)
            print(decay)
            decay_eff = np.clip(decay, 0.0, 15.0)
            print(decay_eff)
            print("debug 1.13.4")
            wid = max(abs(wid), 1e-6)
            print(wid)
            print("debug 1.13.5")
            x_min, x_max = calculate_gaus_extension_limits(cen, wid, decay_eff, factor=2)#3)
            print(x_min)
            print(x_max)
            print("debug 1.13.6")
            # Use decay extrapolator
            best_x, best_fit_y = extrapolate_gaussian_decay(
                amp, cen, wid, decay_eff, x_min, x_max, step=1e-4)
            print("debug 1.13.7")
            print("best_x", best_x)
            print("best_fit_y_max", np.max(best_fit_y))
            print("best_fit_y", best_fit_y)
            new_ind_peak = (np.abs(best_x - x_full[ind_peak])).argmin()
            print("debug 1.13.8")
            left_boundary, right_boundary = calculate_boundaries_acceleration(
                best_x, best_fit_y, new_ind_peak, smoothing_params, pk_sns)
            print("debug 1.13.9")
            print(left_boundary)
            print(right_boundary)
            best_x = best_x[left_boundary - 1 : right_boundary + 1]
            print("debug 1.13.10")
            print(best_x)
            best_fit_y = best_fit_y[left_boundary - 1 : right_boundary + 1]
            print("debug 1.13.11")
            print(best_fit_y)
            area_ensemble.append(simpson(y=best_fit_y, x=best_x))
            print("debug 1.13.12")
    return np.median(area_ensemble), area_ensemble
    
def debug_param_distribution(mu, cov, n_draw=5000):
    """
    Sample from N(mu, cov) and plot:
      1) joint scatter of (wid, decay)
      2) histogram of wid
      3) histogram of decay
    """
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    
    # draw a big batch
    batch = np.random.multivariate_normal(mu, cov, size=n_draw)
    wid   = batch[:,2]
    decay = batch[:,3]
    
    # print("Mean wid, decay:", mu[2], mu[3])
    # print("Std  wid, decay:", np.sqrt(cov[2,2]), np.sqrt(cov[3,3]))
    # print("   Sampled wid min/max:", wid.min(), wid.max())
    # print(" Sampled decay min/max:", decay.min(), decay.max())
    
    # 1) Joint scatter
    plt.figure()
    plt.scatter(wid, decay, alpha=0.2)
    plt.axvline(0)
    plt.axhline(0)
    plt.xlabel("wid")
    plt.ylabel("decay")
    plt.title("Joint draw of (wid, decay)")
    plt.show()
    
    # 2) wid histogram
    plt.figure()
    plt.hist(wid, bins=50)
    plt.xlabel("wid")
    plt.title("Histogram of wid")
    plt.show()
    
    # 3) decay histogram
    plt.figure()
    plt.hist(decay, bins=50)
    plt.xlabel("decay")
    plt.title("Histogram of decay")
    plt.show()
    
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

def estimate_initial_decay(x, y, center_idx):
    left_half = y[:center_idx]
    right_half = y[center_idx:]
    left_slope = np.mean(np.gradient(left_half))
    right_slope = np.mean(np.gradient(right_half))
    asymmetry = right_slope - left_slope

    # Empirical mapping to decay (tweak this based on real data behavior)
    decay_est = np.clip(0.5 * asymmetry, 0.01, 1.5)
    return decay_est

def multigaussian( x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        cen = params[i + 1]
        wid = params[i + 2]
        y += amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))
    return y

def skewed_gaussian(x, amp, cen, sigma, alpha):
    """
    Skewed Gaussian (Skew-Normal) distribution:
    - alpha = 0 gives symmetric Gaussian
    - alpha > 0 → right skew
    - alpha < 0 → left skew
    """
    z = (x - cen) / (sigma * np.sqrt(2))
    return amp * np.exp(-z**2) * (1 + erf(alpha * z))

def gaussian_decay( x, amp, cen, wid, dec):
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2)) * np.exp(-dec * abs(x - cen))

def forward_derivative(x, y):
    fd = np.diff(y) / np.diff(x)
    x_n = x#[:-1]
    return fd, x_n

class FIDAnalyzer:
    def __init__(self, df, window_bounds, gaus_iterations, sample_name, is_reference, max_peaks, sw, sf, pk_sns, pk_pr, max_PA, reference_peaks=None):
        self.fig, self.axs = None, None
        self.df = df
        self.window_bounds = window_bounds
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
    
def run_peak_integrator(data, key, gi, pk_sns, smoothing_params, max_peaks_for_neighborhood, fp, gaussian_fit_mode):
    # Setup data
    xdata = pd.Series(data['Samples'][key]['Raw Data'][data['Integration Metadata']['time_column']])
    ydata = pd.Series(data['Samples'][key]['Raw Data'][data['Integration Metadata']['signal_column']])
    
    # Subset to reference sample
    # --- Subset to global x-limits based on reference sample ---
    peak_times = list(data['Integration Metadata']['peak dictionary'].values())
    rt_buffer = 0.5  # 30 seconds = 0.5 minutes (you suggested 0.4, which is ~24s)
    
    xmin = min(peak_times) - rt_buffer
    xmax = max(peak_times) + rt_buffer
    mask = (xdata >= xmin) & (xdata <= xmax)
    
    xdata = xdata[mask].reset_index(drop=True)
    ydata = ydata[mask].reset_index(drop=True)
    
    # Signal processing
    ydata = smoother(ydata, smoothing_params[0], smoothing_params[1])
    ydata = pd.Series(ydata, index=xdata.index)
    ydata[ydata<0] = 0
    peak_timing = data['Integration Metadata']['peak dictionary'].values()
    data['Samples'][key]['Processed Data'] = {}
    
    base, min_peak_amp = baseline(xdata, ydata, deg=5, max_it=1000, tol=1e-4)
    y_bcorr = ydata-base
    y_bcorr = np.clip(ydata - base, 0, None)
    peak_indices, peak_properties = find_peaks(y_bcorr, height=min_peak_amp, prominence=0.001)
    used_peaks = set()
    matched_indices = []
    presence_flags = []
    
    for pt in peak_timing:
        # Find candidate matches within tolerance
        distances = np.abs(xdata.iloc[peak_indices] - pt)
        candidates = [(idx, dist) for idx, dist in zip(peak_indices, distances) if dist <= 5/60]
    
        # Sort by closeness
        candidates.sort(key=lambda x: x[1])
    
        # Find the closest unused one
        selected = None
        for idx, dist in candidates:
            if idx not in used_peaks:
                selected = idx
                used_peaks.add(idx)
                break
    
        if selected is not None:
            matched_indices.append(selected)
            presence_flags.append(True)
        else:
            matched_indices.append(None)
            presence_flags.append(False)
    
    matched_indices = list(matched_indices)

    fig = plt.figure()
    plt.plot(xdata, y_bcorr, c= 'k', linewidth=1, linestyle='-', zorder=2)
    valleys = find_valleys(y_bcorr, peak_indices)
    peak_labels = list(data['Integration Metadata']['peak dictionary'])
    for label, peak_idx in zip(peak_labels, matched_indices):
        if peak_idx is None:          # in case some peaks weren’t matched
            data['Samples'][key]['Processed Data'][label] = [np.nan]
            continue
        try:
            if gaussian_fit_mode in {"multi", "both"}:
                A, B, peak_neighborhood = find_peak_neighborhood_boundaries(
                    x=xdata, y_smooth=y_bcorr, peaks=peak_indices, valleys=valleys,
                    peak_idx=peak_idx, max_peaks=max_peaks_for_neighborhood,
                    peak_properties=peak_properties, gi=gi,
                    smoothing_params=smoothing_params, pk_sns=pk_sns)
            else:
                peak_neighborhood = [peak_idx]
        
            x_fit, y_fit_smooth, area_smooth, area_ensemble, model_parameters = fit_gaussians(
                xdata, y_bcorr, peak_idx, peak_neighborhood,
                smoothing_params, pk_sns, gi=gi, mode=gaussian_fit_mode)
            plt.fill_between(x_fit, 0, y_fit_smooth, color="red", alpha=0.5, zorder=1)
            
            # Label
            x_peak_label = x_fit[np.argmax(y_fit_smooth)]
            y_peak_label = max(y_fit_smooth)
            plt.text(x_peak_label, y_peak_label * 1.05, label,
            ha='center', va='bottom',
            fontsize=8, color='black', rotation=0,
            zorder=2, bbox=dict(facecolor='white', edgecolor='none', alpha=0))
            # plt.axhline(0, c = 'k')
            
            # Assign data to output
            data['Samples'][key]['Processed Data'][label] = {
                 'Peak Area - median': np.median(area_ensemble),
                 'Peak Area - mean': np.mean(area_ensemble),
                 'Peak Area - standard deviation': np.std(area_ensemble, ddof=1),
                 'Peak Area - number of ensemble members': len(area_ensemble),
                 'Model Parameters': model_parameters,
                 'Retention Time': float(x_peak_label)}
        except Exception as e:
            tqdm.write(f"[Warning] Failed to fit {label} in {key}: {e}")
            data['Samples'][key]['Processed Data'][label] = [np.nan]
        
    
    peak_times = list(data['Integration Metadata']['peak dictionary'].values())
    mean_val = np.mean(peak_times)
    xmin = min(peak_times) - mean_val * 0.1
    xmax = max(peak_times) + mean_val * 0.1
    
    # new y max
    mask = (xdata >= xmin) & (xdata <= xmax)
    y_max = ydata[mask].max()
    plt.xlim(xmin, xmax)
    plt.ylim(0, y_max+y_max*0.1)
    plt.ylabel(data['Integration Metadata']['signal_column'])
    plt.xlabel(data['Integration Metadata']['time_column'])
    plt.savefig(str(fp)+f"/{key}.png", dpi=300)
    plt.close()
    return data


