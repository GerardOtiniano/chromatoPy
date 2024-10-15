# src/chromatopy/chromatoPy.py
from .hplc_integration import hplc_integration
from .assign_indices import assign_indices
from .hplc_to_csv import hplc_to_csv

if __name__ == "hplc_integration":
    hplc_integration()

if __name__ == "hplc_to_csv":
    hplc_to_csv()

if __name__ == "assign_indices":
    assign_indices()
