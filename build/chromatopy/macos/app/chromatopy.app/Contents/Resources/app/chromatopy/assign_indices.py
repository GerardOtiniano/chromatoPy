# src/chromatopy/assign_indices.py
import os
import pandas as pd
from .utils.compounds import *
from .utils.calculate_indices import *


def assign_indices():
    """
    Calculates fractional abundances, methylation, cyclization sets (Raberg et al. 2021), and common indices for brGDGTs.

    The function performs the following tasks:
    1. Prompts the user to input the location of the integrated data (CSV file).
    2. Reads the chromatographic data from the input file.
    3. Calculates the fractional abundance for brGDGTs.
    4. Computes methylation and cyclization sets based on Raberg et al. 2021.
    5. Calculates common indices for the brGDGTs based on the results from previous steps.
    6. Saves the resulting fractional abundance, methylation set, cyclization set, and indices dataframes as separate CSV files in the same directory as the input file.

    Outputs:
    - "chromatopy_indices.csv": CSV file containing calculated indices.
    - "chromatopy_fractional_abundance.csv": CSV file containing fractional abundances of brGDGTs.
    - "chromatopy_meth_set.csv": CSV file containing methylation set data.
    - "chromatopy_cyc_set.csv": CSV file containing cyclization set data.

    Notes
    -----
    - The input file must contain integrated chromatographic data for brGDGTs in CSV format.
    - The Raberg et al. 2021 method is used for calculating methylation and cyclization sets.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function saves the results to CSV files but does not return any values.
    """
    df_path = input("Enter location of integrated data: ")
    df = pd.read_csv(df_path)
    df_fa = calculate_fa(df)
    df_meth, df_cyc = calculate_raberg2021(df)
    df_out = calculate_indices(df_fa, df_meth, df_cyc)
    # setup saving dataframe
    directory_path = os.path.dirname(df_path)
    df_out.to_csv(os.path.join(directory_path, "chromatopy_indices.csv"), index=False)
    df_fa.to_csv(os.path.join(directory_path, "chromatopy_fractional_abundance.csv"), index=False)
    df_meth.to_csv(os.path.join(directory_path, "chromatopy_meth_set.csv"), index=False)
    df_cyc.to_csv(os.path.join(directory_path, "chromatopy_cyc_set.csv"), index=False)
