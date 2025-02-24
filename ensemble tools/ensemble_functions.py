# %% Mean peak area
import os
import json
import glob
import numpy as np

indv_samples = '/Users/gerard/Desktop/UB/GDGT Raw Data/Chromatopy/combined dataset for manuscript'

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
    # If the ensemble is empty, return zeros.
    if data.size == 1:
        return {"mean": 0, "lower_ci": 0, "upper_ci": 0}
    
    boot_means = []
    n = len(data)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    
    mean_val = np.mean(data)
    # lower_bound = np.percentile(boot_means, (100 - ci) / 2)
    # upper_bound = np.percentile(boot_means, 100 - (100 - ci) / 2)
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
                        print(gdgt_info['area_ensemble'])
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
        # First, compute the mean and standard error for each GDGT.
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
    
        # Total mean over all GDGTs.
        T = sum(means.values())
        
        results = {}
        # If total is zero, assign all fractional abundances to 0.
        if T == 0:
            for gdgt in gdgt_data.keys():
                results[gdgt] = {
                    "mean_fa": 0,
                    "fa_lower_bound": 0,
                    "fa_upper_bound": 0
                }
            return results
    
        # Compute fractional abundances and propagate uncertainty via the delta method.
        for i, gdgt in enumerate(means.keys()):
            mu_i = means[gdgt]
            f_i = mu_i / T
            se_i = ses[gdgt]
            
            # Compute the partial derivative for mu_i:
            # d(f_i)/d(mu_i) = (T - mu_i) / T^2.
            dfi_dmui = (T - mu_i) / (T**2)
            
            # For j ≠ i, the partial derivative is: d(f_i)/d(mu_j) = -mu_i/T^2.
            # Thus, the contribution from all other GDGTs is:
            var_other = 0
            for other_gdgt, mu_j in means.items():
                if other_gdgt == gdgt:
                    continue
                se_j = ses[other_gdgt]
                dfi_dmuj = -mu_i / (T**2)
                var_other += (dfi_dmuj**2) * (se_j**2)
            
            # Variance for f_i using the delta method:
            var_fi = (dfi_dmui**2) * (se_i**2) + var_other
            std_fi = np.sqrt(var_fi)
            
            # Define bounds as mean_fa ± 2*std_fi.
            results[gdgt] = {
                "mean_fa": f_i,
                "fa_lower_bound": f_i - 2*std_fi,
                "fa_upper_bound": f_i + 2*std_fi
            }
        
        return results
    
    # Get all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    # Process each JSON file
    for file_path in json_files:
        with open(file_path, 'r') as f:
            sample_data = json.load(f)
        
        sample_name = sample_data.get("Sample Name", os.path.basename(file_path))
        
        # Process each group separately: isoGDGTs and brGDGTs.
        for group in ["isoGDGTs", "brGDGTs"]:
            if group in sample_data:
                group_dict = sample_data[group]
                # Build a dictionary with GDGT names and their ensemble_area lists.
                gdgt_values = {}
                for gdgt_key, gdgt_info in group_dict.items():
                    # Use the key "area_ensemble" for the replicate measurements.
                    if isinstance(gdgt_info, dict) and "area_ensemble" in gdgt_info:
                        ensemble = gdgt_info["area_ensemble"]
                        if ensemble and len(ensemble) > 0:
                            # Use the first element if your structure wraps the actual list inside another list.
                            gdgt_values[gdgt_key] = ensemble
                        else:
                            # If the ensemble is empty, assign an empty list.
                            gdgt_values[gdgt_key] = []
                    else:
                        print(f"Warning: 'area_ensemble' not found for {group} {gdgt_key} in sample {sample_name}.")
                
                if gdgt_values:
                    # Compute the fractional abundances (FA) for this group.
                    fa_results = compute_fractional_abundances(gdgt_values)
                    # Save the computed FA values back into the corresponding GDGT's dictionary.
                    for gdgt_key, fa_stats in fa_results.items():
                        if gdgt_key in group_dict and isinstance(group_dict[gdgt_key], dict):
                            group_dict[gdgt_key]["mean_fa"] = fa_stats["mean_fa"]
                            group_dict[gdgt_key]["fa_lower_bound"] = fa_stats["fa_lower_bound"]
                            group_dict[gdgt_key]["fa_upper_bound"] = fa_stats["fa_upper_bound"]
                else:
                    print(f"No valid area_ensemble data found in group {group} for sample {sample_name}.")
        
        # Save the updated sample_data to a new JSON file in the output folder.
        filename = os.path.basename(file_path)
        output_path = os.path.join(folder_path, filename)
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=4)
folder_path = '/Users/gerard/Desktop/UB/GDGT Raw Data/LeLe/chromatopy - raw hplc csv/Output_chromatoPy/Individual Samples'
ensemble_fractional_abundances(folder_path)

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
folder_path = '/Users/gerard/Desktop/UB/GDGT Raw Data/LeLe/chromatopy - raw hplc csv/Output_chromatoPy/Individual Samples'
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
        
        # Create a row dict starting with the sample name.
        row = {}
        sample_name = sample_data.get("Sample Name", os.path.basename(file_path))
        row["Sample Name"] = sample_name
        
        # Process each group: "Reference", "isoGDGTs", "brGDGTs".
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
                        # Compute the statistics.
                        stats = bootstrap_stats(data, n_bootstrap=n_bootstrap, ci=ci, bound_type=bound_type)
                        # Define column names based on gdgt key.
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


folder_path = "/Users/gerard/Desktop/UB/GDGT Raw Data/Chromatopy/combined dataset for manuscript"  # <-- Update this to your JSON files folder
output_csv = os.path.join(folder_path, "/Users/gerard/Desktop/UB/GDGT Raw Data/Chromatopy/combined dataset for manuscript/compiled_mean_peak_areas_updated.csv")

# Compile the data from the JSON files into a DataFrame.
# To use traditional confidence intervals (e.g., 2.5th/97.5th percentiles for 95% CI):
# df_compiled = compile_mean_ci_peak_areas(folder_path, n_bootstrap=100, ci=95, bound_type="ci")

# To use the 5th and 95th percentiles instead, change bound_type:
df_compiled = compile_mean_ci_peak_areas(folder_path, n_bootstrap=100, ci=95, bound_type="percentile")

# Save the DataFrame to CSV.
df_compiled.to_csv(output_csv, index=False)

print("CSV file created:", output_csv)

