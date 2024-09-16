import os
import pandas as pd
from .compounds import *
from .calculate_indices import *


def assign_indices():
    """
    Function to calculate fractional abundances as well as the methylation and cyclizaiton sets (Raberg et al. 2021)
    of brGDGTs. Common indices are also calcualted. Resultant fractional abundance, methylation, cyclization, and indices
    dataframes are saved as seperate .csv files in the location of the input folder.
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
