import numpy as np
from .compounds import *


def calculate_raberg2021(df):
    """
    Calculates methyl and cyclic group indices based on the methodology outlined
    in Raberg et al. (2021). The function processes a given DataFrame to compute 
    fractional abundances of specific compound groups and returns a DataFrame with 
    the calculated methyl and cyclic indices.
    
    Raberg, J.H., Harning, D.J., Crump, S.E., de Wet, G., Blumm, A., Kopf, S., Geirsdóttir, Á., Miller, G.H., Sepúlveda, J., 2021. Revised fractional abundances and warm-season temperatures substantially improve brGDGT calibrations in lake sediments. Biogeosciences 18, 3579–3603.


    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the compound data. Each column should represent
        a specific compound, and the rows should represent individual samples. The 
        function will calculate the indices only for the compounds present in the 
        DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated methyl and cyclic group indices for 
        each sample. The output includes fractional abundances for the processed 
        compound groups based on their availability in the input DataFrame.

    Notes
    -----
    - Compounds missing from the DataFrame will generate warnings, and indices for 
      those groups will be excluded from the final DataFrame.
    - The function assumes that compound groups like "Meth_a", "Meth_b", etc., are 
      defined globally or passed into the function.
    - This method is based on the approach outlined in Raberg et al. (2021) for 
      calculating specific indices from methyl and cyclic groups in brGDGTs.
     """
    def process_group(group_name, compounds):
        present_compounds = [comp for comp in compounds if comp in df.columns]
        missing_compounds = [comp for comp in compounds if comp not in df.columns]

        # Different warnings based on the missing data
        if not present_compounds:
            txt = compound_group_name_conversion[group_name]
            print(f"Notice: No {txt} are present. This group will not be considered.")
            return pd.DataFrame()  # Return empty DataFrame if no compounds are present
        elif missing_compounds:
            txt = compound_group_name_conversion[group_name]
            print(f"Warning: Not all {txt} compounds are present. Missing: {missing_compounds}")

        # Calculate fractional abundances if there are any compounds present
        df_group = df[present_compounds].div(df[present_compounds].sum(axis=1), axis=0)
        return df_group

    # methset
    Meth_a_group = process_group(br_compounds, Meth_a)
    Meth_ap_group = process_group(br_compounds, Meth_ap)
    Meth_b_group = process_group(br_compounds, Meth_b)
    Meth_bp_group = process_group(br_compounds, Meth_bp)
    Meth_c_group = process_group(br_compounds, Meth_c)
    Meth_cp_group = process_group(br_compounds, Meth_cp)
    meth_df = pd.concat([df.iloc[:, 0], Meth_a_group, Meth_ap_group, Meth_b_group, Meth_bp_group, Meth_c_group, Meth_cp_group], axis=1)

    # cyc set
    CI_I_group = process_group(br_compounds, CI_I)
    CI_II_group = process_group(br_compounds, CI_II)
    CI_III_group = process_group(br_compounds, CI_III)
    cyc_df = pd.concat([df.iloc[:, 0], CI_I_group, CI_II_group, CI_III_group], axis=1)
    return meth_df, cyc_df
