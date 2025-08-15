# %% Mean peak area
import os
import json
import glob
import numpy as np

indv_samples = '/path/to/individual/samples'

def bootstrap_stats(data, n_bootstrap=1000, ci=99):
    """
    Performs bootstrap resampling on a list/array of values and returns the mean,
    lower confidence interval, and upper confidence interval.

    If the data is empty, returns 0 for mean, lower_ci, and upper_ci.

    Parameters:
      - data: list or numpy array of replicate measurements.
      - n_bootstrap: Number of bootstrap resamples (default 1000).
      - ci: Confidence interval percentage (default 95).

    Returns:
      - A dictionary with keys 'mean', 'lower_ci', and 'upper_ci'.
    """
    data = np.array(data)
    if data.size == 1:
        return {"mean": 0, "lower_ci": 0, "upper_ci": 0}
    boot_means = []
    n = len(data)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    
    mean_val = np.mean(data)
    sigma = sigma = np.std(data)
    lower_bound = mean_val - 2 * sigma
    upper_bound = mean_val + 2 * sigma
    
    return {"mean": mean_val, "lower_ci": lower_bound, "upper_ci": upper_bound}

def mean_ci_pa(folder_path):
    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    # Process each JSON file
    for file_path in json_files:
        with open(file_path, 'r') as f:
            sample_data = json.load(f)
        
        # Process the isoGDGTs group if present
        for x in ["Reference", "isoGDGTs", "brGDGTs"]:
            if x in sample_data:
                for gdgt_key, gdgt_info in sample_data[x].items():
                    if isinstance(gdgt_info, dict) and "area_ensemble" in gdgt_info:
                        print(gdgt_info)
                        stats = bootstrap_stats(gdgt_info["area_ensemble"][0], n_bootstrap=1000, ci=95)
                        # Save the computed statistics into the dictionary for this GDGT
                        gdgt_info["mean"] = stats["mean"]
                        gdgt_info["lower_ci"] = stats["lower_ci"]
                        gdgt_info["upper_ci"] = stats["upper_ci"]
                        # gdgt_info["area_ensemble"] = gdgt_info["area_ensemble"][0]
                    else:
                        print(f"Warning: 'area_ensemble' not found for isoGDGT {gdgt_key} in sample {sample_data.get('Sample Name', file_path)}.")
            
        # Save the updated dictionary back to a new JSON file in the output folder
        filename = os.path.basename(file_path)
        # output_path = os.path.join(output_folder, filename)
        output_path = os.path.join(folder_path, filename)
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=4)
mean_ci_pa(indv_samples)


# %% Mean fractional abundance 
import os
import json
import glob
import numpy as np
def ensemble_fractional_abundances(folder_path):
    import os
    import json
    import glob
    import numpy as np
    
    def compute_fractional_abundances(gdgt_data):
        """
        Computes fractional abundances (FA) for a set of GDGT measurements within one group
        (e.g. isoGDGTs or brGDGTs) using the delta method for error propagation.
        
        For each GDGT:
          - The mean (μ) is computed from the ensemble_area values.
          - The standard error (SE) is computed as the sample standard deviation divided by √n.
          - The fractional abundance is f_i = μ_i / T, where T = Σ μ_j.
          - The variance on f_i is estimated via the delta method:
                Var(f_i) = ((T - μ_i)/T²)² * (SE_i)² + Σ_{j≠i} ((μ_i)/T²)² * (SE_j)².
          - The uncertainty is then given as ±2 standard errors (2σ) from the delta method.
        
        Parameters:
          - gdgt_data: dict where keys are GDGT names and values are lists of ensemble_area values.
        
        Returns:
          - A dictionary mapping each GDGT to a dictionary with keys:
              "mean_fa", "fa_lower_bound", and "fa_upper_bound".
        """
        means = {}
        ses = {}
        for gdgt, values in gdgt_data.items():
            data_arr = np.array(values)
            if data_arr.size == 0:
                means[gdgt] = 0
                ses[gdgt] = 0
            else:
                n = len(data_arr)
                mu = np.mean(data_arr)
                # If only one value, we set the standard error to 0.
                sigma = np.std(data_arr, ddof=1) if n > 1 else 0
                sigma = sigma*2
                se = sigma / np.sqrt(n) if n > 1 else 0
                means[gdgt] = mu
                ses[gdgt] = se
    
        T = sum(means.values())
        
        results = {}
        if T == 0:
            for gdgt in gdgt_data.keys():
                results[gdgt] = {
                    "mean_fa": 0,
                    "fa_lower_bound": 0,
                    "fa_upper_bound": 0
                }
            return results
        for i, gdgt in enumerate(means.keys()):
            mu_i = means[gdgt]
            f_i = mu_i / T
            se_i = ses[gdgt]
            
            # Compute the partial derivative for mu_i:
            # d(f_i)/d(mu_i) = (T - mu_i) / T^2.
            dfi_dmui = (T - mu_i) / (T**2)
            var_other = 0
            for other_gdgt, mu_j in means.items():
                if other_gdgt == gdgt:
                    continue
                se_j = ses[other_gdgt]
                dfi_dmuj = -mu_i / (T**2)
                var_other += (dfi_dmuj**2) * (se_j**2)
            var_fi = (dfi_dmui**2) * (se_i**2) + var_other
            std_fi = np.sqrt(var_fi)
            results[gdgt] = {
                "mean_fa": f_i,
                "fa_lower_bound": f_i - 2*std_fi,
                "fa_upper_bound": f_i + 2*std_fi
            }
        return results
    
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            sample_data = json.load(f)
        
        sample_name = sample_data.get("Sample Name", os.path.basename(file_path))
        
        for group in ["isoGDGTs", "brGDGTs"]:
            if group in sample_data:
                group_dict = sample_data[group]
                gdgt_values = {}
                for gdgt_key, gdgt_info in group_dict.items():
                    if isinstance(gdgt_info, dict) and "area_ensemble" in gdgt_info:
                        ensemble = gdgt_info["area_ensemble"]
                        if ensemble and len(ensemble) > 0:
                            gdgt_values[gdgt_key] = ensemble
                        else:
                            gdgt_values[gdgt_key] = []
                    else:
                        print(f"Warning: 'area_ensemble' not found for {group} {gdgt_key} in sample {sample_name}.")
                
                if gdgt_values:
                    fa_results = compute_fractional_abundances(gdgt_values)
                    for gdgt_key, fa_stats in fa_results.items():
                        if gdgt_key in group_dict and isinstance(group_dict[gdgt_key], dict):
                            group_dict[gdgt_key]["mean_fa"] = fa_stats["mean_fa"]
                            group_dict[gdgt_key]["fa_lower_bound"] = fa_stats["fa_lower_bound"]
                            group_dict[gdgt_key]["fa_upper_bound"] = fa_stats["fa_upper_bound"]
                else:
                    print(f"No valid area_ensemble data found in group {group} for sample {sample_name}.")
        filename = os.path.basename(file_path)
        output_path = os.path.join(folder_path, filename)
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=4)
folder_path = 'path/to/files'
test = ensemble_fractional_abundances(folder_path)

# %% FA csv output 
import os
import json
import glob
import pandas as pd

def compile_fa_dataframe(folder_path):
    """
    Reads each JSON file in folder_path, extracts the computed fractional abundances 
    (mean, lower, and upper) for each GDGT type, and compiles them into a DataFrame.
    
    Each row corresponds to one sample, and for each GDGT (e.g., 'Ia') three columns are created:
      - Ia (the mean fractional abundance)
      - Ia_lower (the lower confidence bound)
      - Ia_upper (the upper confidence bound)
    
    Parameters
    ----------
    folder_path : str
        The folder containing the JSON files.
    
    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with one row per sample and columns for each GDGT's FA and uncertainty.
    """
    rows = []
    # Get all JSON files
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            sample_data = json.load(f)
        
        # Use sample name from the JSON or the filename as a fallback
        sample_name = sample_data.get("Sample Name", os.path.basename(file_path))
        row = {"Sample Name": sample_name}
        
        # Process each GDGT group (modify groups as needed)
        for group in ["isoGDGTs", "brGDGTs"]:
            if group in sample_data:
                group_dict = sample_data[group]
                for gdgt_key, gdgt_info in group_dict.items():
                    # We expect the computed FA values to have been added by your earlier function.
                    if isinstance(gdgt_info, dict) and "mean_fa" in gdgt_info:
                        row[gdgt_key] = gdgt_info["mean_fa"]
                        row[f"{gdgt_key}_lower"] = gdgt_info["fa_lower_bound"]
                        row[f"{gdgt_key}_upper"] = gdgt_info["fa_upper_bound"]
                    else:
                        # Optionally, warn if expected keys are not found
                        print(f"Warning: FA data not found for {group} {gdgt_key} in sample {sample_name}.")
        rows.append(row)
    
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(rows)
    return df

# Example usage:
folder_path = '/folder/to/individual/sample'
df_fa = compile_fa_dataframe(folder_path)
print(df_fa.head())

# %% Output as csv
import os
import glob
import json
import numpy as np
import pandas as pd

def bootstrap_stats(data, n_bootstrap=1000, ci=95, bound_type="ci"):
    """
    Performs bootstrap resampling on the given data (a list or numpy array) 
    and returns a dictionary with keys: "mean", "lower_ci", and "upper_ci".
    
    Parameters:
      data (list or np.array): Input data.
      n_bootstrap (int): Number of bootstrap samples.
      ci (float): Confidence interval percentage (used if bound_type=='ci').
      bound_type (str): 
          'ci' uses (100-ci)/2 and 100-(100-ci)/2 percentiles for CI limits,
          'percentile' uses the 5th and 95th percentiles.
          
    Returns:
      dict: Dictionary with the mean, lower_ci, and upper_ci.
      If the input data is empty, returns zeros.
    """
    data = np.array(data)
    if data.size == 0:
        return {"mean": 0, "lower_ci": 0, "upper_ci": 0}
    
    boot_means = []
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    mean_val = np.mean(data)
    
    if bound_type == "ci":
        lower_bound = np.percentile(boot_means, (100 - ci) / 2)
        upper_bound = np.percentile(boot_means, 100 - (100 - ci) / 2)
    elif bound_type == "percentile":
        lower_bound = np.percentile(boot_means, 5)
        upper_bound = np.percentile(boot_means, 95)
    else:
        raise ValueError("Invalid bound_type specified. Use 'ci' or 'percentile'.")
    
    return {"mean": mean_val, "lower_ci": lower_bound, "upper_ci": upper_bound}

def compile_mean_ci_peak_areas(folder_path, n_bootstrap=1000, ci=95, bound_type="ci"):
    """
    Processes all JSON files in folder_path and compiles a CSV file
    with the sample name in the first column and, for each GDGT type found in
    groups "Reference", "isoGDGTs", and "brGDGTs", the mean peak area and its
    lower and upper limits as calculated by bootstrap_stats.

    The output CSV will have columns named:
      - Sample Name
      - <GDGT> (mean)
      - <GDGT>_lower_ci
      - <GDGT>_upper_ci

    Parameters:
      folder_path (str): Folder containing the JSON files.
      n_bootstrap (int): Number of bootstrap samples.
      ci (float): Confidence interval percentage (used if bound_type=='ci').
      bound_type (str): 'ci' for confidence intervals, 'percentile' for 5th and 95th percentiles.
      
    Returns:
      DataFrame: A Pandas DataFrame with the compiled data.
    """
    # List to hold one dictionary per sample.
    rows = []
    # Find all JSON files in the folder.
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            sample_data = json.load(f)
        row = {}
        sample_name = sample_data.get("Sample Name", os.path.basename(file_path))
        row["Sample Name"] = sample_name
        for group in ["Reference", "isoGDGTs", "brGDGTs"]:
            if group in sample_data:
                for gdgt_key, gdgt_info in sample_data[group].items():
                    if isinstance(gdgt_info, dict) and "area_ensemble" in gdgt_info:
                        ensemble = gdgt_info["area_ensemble"]
                        # Check if ensemble is non-empty; assume it might be stored as [list] or directly a list.
                        if ensemble and len(ensemble) > 0:
                            # If the first element is itself a list, use it; otherwise, use ensemble.
                            if isinstance(ensemble[0], list):
                                data = ensemble[0]
                            else:
                                data = ensemble
                        else:
                            data = []
                        stats = bootstrap_stats(data, n_bootstrap=n_bootstrap, ci=ci, bound_type=bound_type)
                        col_mean  = f"{gdgt_key}"
                        col_lower = f"{gdgt_key}_lower_ci"
                        col_upper = f"{gdgt_key}_upper_ci"
                        row[col_mean]  = stats["mean"]
                        row[col_lower] = stats["lower_ci"]
                        row[col_upper] = stats["upper_ci"]
                    else:
                        print(f"Warning: 'area_ensemble' not found for {group} {gdgt_key} in sample {sample_name}.")
        
        rows.append(row)
    
    # Create a DataFrame from the list of rows.
    df = pd.DataFrame(rows)
    return df


folder_path = '/Users/gerard/Desktop/UB/GDGT Raw Data/Chromatopy/combined dataset for manuscript/individual samples'
output_csv = os.path.join(folder_path, "output_TEST.csv")
df_compiled = compile_mean_ci_peak_areas(folder_path, n_bootstrap=100, ci=100, bound_type="percentile")
df_compiled.to_csv(output_csv, index=False)

# %%
import os
import glob
import json
import numpy as np
import pandas as pd

def minmax_stats(data):
    """
    Returns the mean, min, and max of the given data.
    
    Parameters:
      data (list or np.array): Input data.
      
    Returns:
      dict: {"mean": mean_val, "lower_ci": min_val, "upper_ci": max_val}
    """
    data = np.array(data)
    if data.size == 0:
        return {"mean": 0, "lower_ci": 0, "upper_ci": 0}
    
    mean_val = np.mean(data)
    lower_bound = np.min(data)
    upper_bound = np.max(data)
    
    return {"mean": mean_val, "lower_ci": lower_bound, "upper_ci": upper_bound}

def compile_mean_minmax_peak_areas(folder_path):
    """
    Processes all JSON files in folder_path and compiles a CSV file
    with the sample name in the first column and, for each GDGT type found in
    groups "Reference", "isoGDGTs", and "brGDGTs", the mean peak area and its
    min and max values from the ensemble.
    """
    rows = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            sample_data = json.load(f)
        row = {}
        sample_name = sample_data.get("Sample Name", os.path.basename(file_path))
        row["Sample Name"] = sample_name
        
        for group in ["Reference", "isoGDGTs", "brGDGTs"]:
            if group in sample_data:
                for gdgt_key, gdgt_info in sample_data[group].items():
                    if isinstance(gdgt_info, dict) and "area_ensemble" in gdgt_info:
                        ensemble = gdgt_info["area_ensemble"]
                        if ensemble and len(ensemble) > 0:
                            if isinstance(ensemble[0], list):
                                data = ensemble[0]
                            else:
                                data = ensemble
                        else:
                            data = []
                        
                        stats = minmax_stats(data)
                        row[gdgt_key] = stats["mean"]
                        row[f"{gdgt_key}_lower_ci"] = stats["lower_ci"]
                        row[f"{gdgt_key}_upper_ci"] = stats["upper_ci"]
                    else:
                        print(f"Warning: 'area_ensemble' not found for {group} {gdgt_key} in sample {sample_name}.")
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

# Example usage:
folder_path = '/Users/gerard/Desktop/UB/GDGT Raw Data/Chromatopy/combined dataset for manuscript/individual samples'
output_csv = os.path.join(folder_path, "output_TEST2.csv")
df_compiled = compile_mean_minmax_peak_areas(folder_path)
df_compiled.to_csv(output_csv, index=False)


# %%
import os
import glob
import json
import numpy as np
import pandas as pd

def minmax_error_stats(data):
    """
    Returns the mean, and the differences between the mean and min/max values.
    
    Parameters:
      data (list or np.array): Input data.
      
    Returns:
      dict: {"mean": mean_val, "lower_ci": mean-min_val, "upper_ci": max_val-mean}
    """
    data = np.array(data)
    if data.size == 0:
        return {"mean": 0, "lower_ci": 0, "upper_ci": 0}
    
    mean_val = np.mean(data)
    lower_err = mean_val - np.min(data)
    upper_err = np.max(data) - mean_val
    
    return {"mean": mean_val, "lower_ci": lower_err, "upper_ci": upper_err}

def compile_mean_minmax_errors(folder_path):
    """
    Processes all JSON files in folder_path and compiles a CSV file
    with the sample name in the first column and, for each GDGT type found in
    groups "Reference", "isoGDGTs", and "brGDGTs", the mean peak area and
    the differences from mean to min (lower_ci) and mean to max (upper_ci).
    """
    rows = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            sample_data = json.load(f)
        row = {}
        sample_name = sample_data.get("Sample Name", os.path.basename(file_path))
        row["Sample Name"] = sample_name
        
        for group in ["Reference", "isoGDGTs", "brGDGTs"]:
            if group in sample_data:
                for gdgt_key, gdgt_info in sample_data[group].items():
                    if isinstance(gdgt_info, dict) and "area_ensemble" in gdgt_info:
                        ensemble = gdgt_info["area_ensemble"]
                        if ensemble and len(ensemble) > 0:
                            if isinstance(ensemble[0], list):
                                data = ensemble[0]
                            else:
                                data = ensemble
                        else:
                            data = []
                        
                        stats = minmax_error_stats(data)
                        row[gdgt_key] = stats["mean"]
                        row[f"{gdgt_key}_lower_ci"] = stats["lower_ci"]
                        row[f"{gdgt_key}_upper_ci"] = stats["upper_ci"]
                    else:
                        print(f"Warning: 'area_ensemble' not found for {group} {gdgt_key} in sample {sample_name}.")
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

# Example usage:
# folder_path = '/Users/gerard/Desktop/UB/GDGT Raw Data/Chromatopy/combined dataset for manuscript/individual samples'
folder_path = '/Users/gerard/Documents/GitHub/chromatoPy_manuscript/Data/Sudip Comparison/Chromatopy-selected/Output_chromatoPy_isoGDGTs/Individual Samples'
output_csv = os.path.join(folder_path, "output_TEST_sud_iso.csv")
df_compiled = compile_mean_minmax_errors(folder_path)
df_compiled.to_csv(output_csv, index=False)

# %%
import os
import glob
import json
import numpy as np
import pandas as pd

# def percentile95_error_stats(data):
#     """
#     Returns the mean, and the differences between the mean and the
#     2.5th / 97.5th percentiles of the data.
    
#     Parameters:
#       data (list or np.array): Input data.
      
#     Returns:
#       dict: {"mean": mean_val, "lower_ci": mean - p2.5, "upper_ci": p97.5 - mean}
#     """
#     data = np.array(data)
#     if data.size == 0:
#         return {"mean": 0, "lower_ci": 0, "upper_ci": 0}
    
#     mean_val = np.mean(data)
#     lower_bound = np.percentile(data, 2.5)
#     upper_bound = np.percentile(data, 97.5)
    
#     lower_err = mean_val - lower_bound
#     upper_err = upper_bound - mean_val
    
#     return {"mean": mean_val, "lower_ci": lower_err, "upper_ci": upper_err}
def percentile95_to_sigma_stats(data):
    """
    Returns the mean and the 1σ-equivalent lower/upper errors
    derived from the 95% confidence interval (2.5th and 97.5th percentiles),
    ignoring values < 1.

    Parameters:
      data (list or np.array): Input data.

    Returns:
      dict: {"mean": mean_val,
             "lower_sigma": lower_err_sigma,
             "upper_sigma": upper_err_sigma}
    """
    print_tick=False
    # Convert to numpy array and filter out values < 1
    data = np.array(data, dtype=float)
    if np.any(data < 0):
        print_tick = True
    data = data[data >= 0]

    # Handle empty case
    if data.size == 0:
        return {"mean": 0.0, "lower_sigma": 0.0, "upper_sigma": 0.0}, print_tick
    
    mean_val = np.nanmedian(data)
    median_val = np.nanmedian(data)
    lower_bound = np.nanpercentile(data, 2.5)
    # print(lower_bound)
    
    upper_bound = np.nanpercentile(data, 97.5)
    # print(upper_bound)
    
    lower_err_95 = mean_val - lower_bound
    upper_err_95 = upper_bound - mean_val

    # Here we keep the 95% CI half-widths directly (not converting to actual sigma scaling factor)
    out_dict = {
        "mean": mean_val,
        "lower_sigma": lower_err_95,
        "upper_sigma": upper_err_95}
    return out_dict, print_tick

def compile_mean_percentile95_errors(folder_path):
    """
    Processes all JSON files in folder_path and compiles a CSV file
    with the sample name in the first column and, for each GDGT type found in
    groups "Reference", "isoGDGTs", and "brGDGTs", the mean peak area and
    the differences from mean to the 2.5th and 97.5th percentiles.
    """
    rows = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            sample_data = json.load(f)
        row = {}
        sample_name = sample_data.get("Sample Name", os.path.basename(file_path))
        row["Sample Name"] = sample_name
        for group in ["Reference", "isoGDGTs", "brGDGTs"]:

            if group in sample_data:
                for gdgt_key, gdgt_info in sample_data[group].items():
                    if isinstance(gdgt_info, dict) and "area_ensemble" in gdgt_info:
                        ensemble = gdgt_info["area_ensemble"]
                        if sample_name=="H1801000119":
                            if gdgt_key=="IIb'":
                                print(gdgt_key)
                                print(ensemble)
                        if ensemble and len(ensemble) > 0:
                            if isinstance(ensemble[0], list):
                                data = ensemble[0]
                            else:
                                data = ensemble
                        else:
                            data = []
                        stats, print_tick = percentile95_to_sigma_stats(data)
                        row[gdgt_key] = stats["mean"]
                        row[f"{gdgt_key}_lower_ci"] = stats["lower_sigma"]
                        row[f"{gdgt_key}_upper_ci"] = stats["upper_sigma"]
                    else:
                        print(f"Warning: 'area_ensemble' not found for {group} {gdgt_key} in sample {sample_name}.")
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

# Example usage:
folder_path = '/Users/gerard/Desktop/redo/isogdgts/Largest peak/Output_chromatoPy/Individual Samples'
output_csv = os.path.join(folder_path, "output_FINAL.csv")
df_compiled = compile_mean_percentile95_errors(folder_path)
df_compiled.to_csv(output_csv, index=False)


