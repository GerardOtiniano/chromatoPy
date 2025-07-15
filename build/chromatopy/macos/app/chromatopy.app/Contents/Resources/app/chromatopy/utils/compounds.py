# src/chromatopy/utils/compounds.py
"""
Compound dictinoaries for calculating indices.
"""

# Traditional fractional abundance grouping
br_compounds = ["IIIa", "IIIa'", "IIIb", "IIIb'", "IIIc", "IIIc'", "IIa", "IIa'", "IIb", "IIb'", "IIc", "IIc'", "Ia", "Ib", "Ic"]
tetra = ["Ia", "Ib", "Ic"]
penta = ["IIa", "IIa'", "IIb", "IIb'", "IIc", "IIc'"]
hexa = ["IIIa", "IIIa'", "IIIb", "IIIb'", "IIIc", "IIIc'"]

iso_compounds = ["GDGT-0", "GDGT-1", "GDGT-2", "GDGT-3", "GDGT-4", "GDGT-4'"]
oh_compounds = ["OH-GDGT-0", "OH-GDGT-1", "2OH-GDGT-0", "OH-GDGT-2"]
compound_group_name_conversion = {"oh_compounds": "OH-GDGTs", "br_compounds": "brGDGTs", "iso_compounds": "isoGDGTs"}


# Raberg et al. 2021 Cyclization group
CI_I = ["Ia", "Ib", "Ic"]
CI_II = ["IIa", "IIa'", "IIb", "IIb'", "IIc", "IIc'"]
CI_III = ["IIIa", "IIIa'", "IIIb", "IIIb'", "IIIc", "IIIc'"]

# Raberg et al. 2021 Methylation group
Meth_a = ["Ia", "IIa", "IIIa"]
Meth_ap = ["IIa'", "IIIa'"]
Meth_b = ["Ib", "IIb", "IIIb"]
Meth_bp = ["IIb'", "IIIb'"]
Meth_c = ["Ic", "IIc", "IIIc"]
Meth_cp = ["IIc'", "IIIc'"]
