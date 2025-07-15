# src/chromatopy/utils/calculate_indices.py

import numpy as np
import pandas as pd
from .compounds import *

def calculate_fa(df):
    """
    Calculates fractional abundances for different GDGT compound groups (brGDGTs, isoGDGTs, and OH-GDGTs) based on
    the provided dataframe of peak areas. Fractional abundances for each group are computed by dividing the peak
    areas of individual compounds by the total peak area for that group.
    
    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame containing chromatographic peak areas. Each column represents a compound, and each row
        represents a sample. The first column is assumed to contain sample identifiers (e.g., sample names).
    
    Returns
    -------
    result_df : DataFrame
        A pandas DataFrame with the fractional abundances for each compound group (brGDGTs, isoGDGTs, OH-GDGTs), 
        merged with the original sample identifiers.
    
    Notes
    -----
    - Missing compounds will result in warnings, but the function will still calculate fractional abundances
      for any compounds present in the group.
    - If no compounds are present for a group, the group will be excluded from the result.
    """
    # Function to process each compound group
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

    # Analyzing each group
    br_group = process_group("br_compounds", br_compounds)
    iso_group = process_group("iso_compounds", iso_compounds)
    oh_group = process_group("oh_compounds", oh_compounds)

    # Merging fractional dataframes back with sample names
    result_df = pd.concat([df.iloc[:, 0], br_group, iso_group, oh_group], axis=1)
    return result_df


def calculate_indices(df_fa, meth_df, cyc_df):
    """
    Calculates various indices (such as MBT', CBT5Me, CBT', BIT, etc.) for brGDGTs based on fractional abundances
    and methylation/cyclization sets. The function also computes additional indices like conductivity.
    
    Parameters
    ----------
    df_fa : DataFrame
        DataFrame containing fractional abundances of brGDGT compounds.
    meth_df : DataFrame
        DataFrame containing methylation set values.
    cyc_df : DataFrame
        DataFrame containing cyclization set values.
    
    Returns
    -------
    df : DataFrame
        DataFrame containing the calculated indices.
    """
    df = pd.DataFrame()

    # Ensure Sample Name is available
    if "Sample Name" in df_fa:
        df["Sample Name"] = df_fa["Sample Name"]

    # Ternary
    tetra_ = df_fa[tetra].sum(axis=1) if all(col in df_fa for col in tetra) else 0
    penta_ = df_fa[penta].sum(axis=1) if all(col in df_fa for col in penta) else 0
    hexa_ = df_fa[hexa].sum(axis=1) if all(col in df_fa for col in hexa) else 0
    sum_ = df_fa[br_compounds].sum(axis=1) if all(col in df_fa for col in br_compounds) else 1  # Avoid division by zero

    df["%tetra"] = np.where(sum_ != 0, tetra_ / sum_, np.nan)
    df["%penta"] = np.where(sum_ != 0, penta_ / sum_, np.nan)
    df["%hexa"] = np.where(sum_ != 0, hexa_ / sum_, np.nan)

    # Indices
    mbt_denom = df_fa.get("Ia", 0) + df_fa.get("Ib", 0) + df_fa.get("Ic", 0) + df_fa.get("IIa", 0) + df_fa.get("IIb", 0) + df_fa.get("IIc", 0) + df_fa.get("IIIa", 0)
    df["MBT'5Me"] = np.where(mbt_denom != 0, (df_fa.get("Ia", 0) + df_fa.get("Ib", 0) + df_fa.get("Ic", 0)) / mbt_denom, np.nan)

    cbt5_denom = df_fa.get("Ia", 0) + df_fa.get("IIa", 0)
    df["CBT5Me"] = np.where(cbt5_denom != 0, -np.log10((df_fa.get("Ib", 0) + df_fa.get("IIb", 0)) / cbt5_denom), np.nan)

    cbt_prime_denom = df_fa.get("Ia", 0) + df_fa.get("IIa", 0) + df_fa.get("IIIa", 0)
    df["CBT'"] = np.where(cbt_prime_denom != 0, -np.log10((df_fa.get("Ic", 0) + df_fa.get("IIa'", 0) + df_fa.get("IIb'", 0) + df_fa.get("IIc'", 0) + df_fa.get("IIIa'", 0) + df_fa.get("IIIb'", 0) + df_fa.get("IIIc'", 0)) / cbt_prime_denom), np.nan)

    dc_denom = df_fa.get("Ia", 0) + df_fa.get("IIa", 0) + df_fa.get("IIa'", 0) + df_fa.get("Ib", 0) + df_fa.get("IIb", 0) + df_fa.get("IIb'", 0)
    df["DC"] = np.where(dc_denom != 0, (df_fa.get("Ib", 0) + df_fa.get("IIb", 0) + df_fa.get("IIb'", 0)) / dc_denom, np.nan)

    df["cald/cren"] = np.where(df_fa.get("GDGT-4", 1) != 0, df_fa.get("GDGT-0", 0) / df_fa.get("GDGT-4", 1), np.nan)

    hp5_denom = df_fa.get("IIa", 0) + df_fa.get("IIIa", 1)
    df["HP5"] = np.where(hp5_denom != 0, df_fa.get("IIIa", 0) / hp5_denom, np.nan)

    bit_denom = df_fa.get("Ia", 0) + df_fa.get("IIa", 0) + df_fa.get("IIIa", 0) + df_fa.get("GDGT-4", 1)
    df["BIT"] = np.where(bit_denom != 0, (df_fa.get("Ia", 0) + df_fa.get("IIa", 0) + df_fa.get("IIIa", 0)) / bit_denom, np.nan)

    ir_denom = df_fa.get("Ia", 0) + df_fa.get("IIa", 0) + df_fa.get("IIIa", 0) +df_fa.get("Ib", 0) + df_fa.get("IIb", 0) + df_fa.get("IIIb", 0) +df_fa.get("Ic", 0) + df_fa.get("IIc", 0) + df_fa.get("IIIc", 0) +  df_fa.get("IIa'", 0) + df_fa.get("IIIa'", 0) + df_fa.get("IIb'", 0) + df_fa.get("IIIb'", 0) + df_fa.get("IIc'", 0) + df_fa.get("IIIc'", 0)
    ir_numer =  df_fa.get("IIa'", 0) + df_fa.get("IIIa'", 0) + df_fa.get("IIb'", 0) + df_fa.get("IIIb'", 0) + df_fa.get("IIc'", 0) + df_fa.get("IIIc'", 0)
    df['IR6Me'] = np.where(ir_denom !=0, (ir_numer)/(ir_denom))
    
    fc_denom = (
        df_fa.get("Ia", 0)
        + df_fa.get("Ib", 0)
        + df_fa.get("Ic", 0)
        + df_fa.get("IIa", 0)
        + df_fa.get("IIa'", 0)
        + df_fa.get("IIb", 0)
        + df_fa.get("IIb'", 0)
        + df_fa.get("IIIa", 0)
        + df_fa.get("IIIa'", 0)
        + df_fa.get("IIIb", 0)
        + df_fa.get("IIIb'", 0)
        + df_fa.get("IIc", 0)
        + df_fa.get("IIc'", 0)
        + df_fa.get("IIIc", 0)
        + df_fa.get("IIIc'", 0)
    )
    df["fC"] = np.where(fc_denom != 0, 0.5 * ((df_fa.get("Ib", 0) + df_fa.get("IIb", 0) + df_fa.get("IIb'", 0) + df_fa.get("IIIb", 0) + df_fa.get("IIIb'", 0) + 2 * (df_fa.get("Ic", 0) + df_fa.get("IIc", 0) + df_fa.get("IIc'", 0) + df_fa.get("IIIc", 0) + df_fa.get("IIIc'", 0))) / fc_denom), np.nan)

    # Conductivity
    df["conductivity"] = np.exp(6.62 + 8.87 * cyc_df.get("Ib", 0) + 5.12 * cyc_df.get("IIa'", 0) ** 2 + 10.64 * cyc_df.get("IIa", 0) ** 2 - 8.59 * cyc_df.get("IIa", 0) - 4.32 * cyc_df.get("IIIa'", 0) ** 2 - 5.32 * cyc_df.get("IIIa", 0) ** 2 - 142.67 * cyc_df.get("IIIb", 0) ** 2)

    return df

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
