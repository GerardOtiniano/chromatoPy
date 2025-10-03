[![ChromatoPy Logo](misc/chromatoPy.png)](https://github.com/GerardOtiniano/chromatoPy/blob/2b36a74ed639d5c30ae1e143843c1532b0a84237/misc/chromatoPy.png)

# chromatoPy (2.2.0)

chromatoPy is an open-source Python package designed to streamline the integration and analysis of High-Performance Liquid Chromatography (HPLC) and Gas Chromatograph Flame Ionization Detector (GC-FID) data. It features flexible multi-Gaussian and single Gaussian fitting algorithms to detect, fit, and integrate peaks from chromatographic data, enabling efficient analysis and processing of complex datasets. Note, interactive integration requires internal spike standard (Trace 744).

## Features

* **Flexible Gaussian Fitting**: Supports both single and multi‑Gaussian peak fitting algorithms with built‑in uncertainty estimation (area ensembles from parameter variance).
* **Data Integration**: Integrates chromatographic peak data for precise quantification.
* **Customizable Analysis**: Allows for the adjustment of fitting parameters to accommodate various peak shapes.
* **Input Support**: Works with HPLC data converted to `.csv` (via `rainbow-api`), and GC‑FID `.txt` exports from Chromeleon.
* **FID Module**: Comprehensive support for GC‑FID peak integration:

  * **Automatic integration** via `integration(manual_peak_integration=False)` (default).
  * **Manual integration**: user clicks individual peaks with `integration(manual_peak_integration=True)`.
  * **Stored‑label manual mode**: `integration(manual_peak_integration=True, peak_labels=True)` uses a JSON of peak labels + x‑limits.
  * **Peak Label Editor**: `chromatopy.FID.peak_label_editor()` opens a GUI to add/update stored retention‑time windows.
  * **Chromatogram Plotting**: `chromatopy.FID.plot_chromatogram(time_window=[xmin, xmax])` displays all sample traces in the chosen window.
  * **Sample Clustering**: `chromatopy.FID.clusterer()` groups samples by chromatogram similarity (handles RT shifts).
  * **Results I/O**:

    * Output saved automatically as `FID_output.json` in the data directory.
    * Load with `chromatopy.FID.load_results(output_path)`.
    * Delete unwanted samples via `chromatopy.FID.delete_samples(json_path, to_delete=[...])`.


## Installation

To install chromatoPy from the GitHub repository, you can use the following pip command:

```bash
pip install chromatopy
```

## Requirements

* Python 3.12.4 or higher
* Automatic dependencies (installed with pip):

  * numpy==1.26.4
  * pandas==2.2.2
  * scikit‑learn==1.4.2
  * scipy==1.13.1
  * matplotlib==3.8.4
  * rainbow-api==1.0.9
  * pybaselines==1.1.0
  * tqdm==2.0.0
  * pillow==11.0.0
  * imagehash==4.0.0

## Note on Development and Testing

This package has been developed and tested using the Spyder IDE. While it is expected to work in other development environments, it has not been specifically tested with other IDEs. If you encounter any issues when using the package in a different environment, please feel free to raise an issue or reach out for support.

environment‑specific issues.

## Usage

### HPLC workflows

```python
import chromatopy

# Convert raw HPLC to CSV:
chromatopy.hplc_to_csv(input_folder, output_folder)

# Run HPLC integration & index assignment:
chromatopy.hplc_integration()
chromatopy.assign_indices()
```

### GC‑FID Integration (FID module)

```python
import chromatopy
from chromatopy import FID

# 1. Automatic peak integration (default):
FID.integration(folder_path)

# 2. Manual peak picking:
FID.integration(folder_path, manual_peak_integration=True)

# 3. Manual + stored labels:
FID.integration(folder_path, manual_peak_integration=True, peak_labels=True)

# 4. Edit stored peak labels:
chromatopy.FID.peak_label_editor()

# 5. Plot all chromatograms in a time window:
chromatopy.FID.plot_chromatogram(time_window=[0, 20])

# 6. Cluster samples by chromatogram similarity:
chromatopy.FID.clusterer(n_clusters=4)

# 7. Load saved results:
results = chromatopy.FID.load_results(output_path)

# 8. Delete samples from JSON:
chromatopy.FID.delete_samples(json_path, to_delete=["sample1", "sample2"])
```

## Input Data Requirements

* **HPLC**: expects `.csv` outputs from `chromatopy.hplc_to_csv()`.
* **FID**: reads `.txt` files exported from Chromeleon, containing a header line `Chromatogram Data Information:`.

## JSON Output Structure

The `FID_output.json` has two top‑level keys:

1. **`Samples`**: a dict mapping each sample name →

   * `Metadata`: raw file metadata
   * `Raw Data`: original time & signal arrays
   * `Processed Data`: dict of peak labels →

     * `Area Ensembles`: list of calculated peak areas
     * `Model Parameters`: fitted Gaussian params & metadata

2. **`Integration Metadata`**: info on how integration was run:

   * `peak dictionary`: dict (label→RT) or list of labels
   * `x limits`: \[xmin, xmax] (when using stored labels)
   * `time_column` & `signal_column`

## Versioning

Version numbers are reported in an "X.Y.Z" format.

- **X (Major version):** changes that would require the user to adapt their usage of the package (e.g., removing or renaming functions or methods, introducing new functions that change functionality).
- **Y (Minor version):** modifications to functions or new features that are backward-compatible.
- **Z (Patch version):** minor bug fixes or enhancements that do not affect the core interface/method.

## Contributing

Contributions to chromatoPy are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or inquiries, please contact:

- Author: Dr. Gerard Otiniano & Dr. Elizabeth Thomas
- Email: gerardot@buffalo.edu
