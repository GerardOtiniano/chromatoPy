[![ChromatoPy Logo](misc/chromatoPy.png)](https://github.com/GerardOtiniano/chromatoPy/blob/2b36a74ed639d5c30ae1e143843c1532b0a84237/misc/chromatoPy.png)

# chromatoPy (1.0.0)

chromatoPy is an open-source Python package designed to streamline the integration and analysis of High-Performance Liquid Chromatography (HPLC) data. It features flexible multi-Gaussian and single Gaussian fitting algorithms to detect, fit, and integrate peaks from chromatographic data, enabling efficient analysis and processing of complex datasets. Note, interactive integration requires internal spike standard (Trace 744).

## Features

- **Flexible Gaussian Fitting**: Supports both single and multi‑Gaussian peak fitting algorithms with built‑in uncertainty estimation (area ensembles from parameter variance).
- **Data Integration**: Integrates chromatographic peak data for precise quantification.
- **Customizable Analysis**: Allows for the adjustment of fitting parameters to accommodate various peak shapes.
- **Input Support**: Works with HPLC data converted to `.csv` (via `rainbow-api`).

## Installation

## Requirements

## Note on Development and Testing

## Usage

## JSON Output Structure

The `FID_output.json` has two top‑level keys:

1. **`Samples`**: a dict mapping each sample name →

   - `Metadata`: raw file metadata
   - `Raw Data`: original time & signal arrays
   - `Processed Data`: dict of peak labels →

     - `Area Ensembles`: list of calculated peak areas
     - `Model Parameters`: fitted Gaussian params & metadata

2. **`Integration Metadata`**: info on how integration was run:

   - `peak dictionary`: dict (label→RT) or list of labels
   - `x limits`: \[xmin, xmax] (when using stored labels)
   - `time_column` & `signal_column`

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
