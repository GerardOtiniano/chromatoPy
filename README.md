[![ChromatoPy Logo](misc/chromatoPy.png)](https://github.com/GerardOtiniano/chromatoPy/blob/2b36a74ed639d5c30ae1e143843c1532b0a84237/misc/chromatoPy.png)

# chromatoPy (1.8.0)

chromatoPy is an open-source Python package designed to streamline the integration and analysis of High-Performance Liquid Chromatography (HPLC) and Gas Chromatograph Flame Ionization Detector (GC-FID) data. It features flexible multi-Gaussian and single Gaussian fitting algorithms to detect, fit, and integrate peaks from chromatographic data, enabling efficient analysis and processing of complex datasets. Note, interactive integration requires internal spike standard (Trace 744).

## Features

- **Flexible Gaussian Fitting**: Supports both single and multi-Gaussian peak fitting algorithms.
  -- **Fit Uncertainty**: An ensemble peak areas are calculated using uncertainty in fitting parameters (peak width, height, and centre).
- **Data Integration**: Integrates chromatographic peak data for precise quantification.
- **Customizable Analysis**: Allows for the adjustment of fitting parameters to accommodate various peak shapes.
- **Input Support**: Works with HPLC data converted to .csv format using a built in function that utilizes the package "rainbow".

## Installation

To install chromatoPy from the GitHub repository, you can use the following pip command:

```bash
pip install chromatopy
```

## Requirements

- Python 3.12.4 or higher
- Dependencies are automatically installed when you install the package:
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.4.2
- scipy==1.13.1
- matplotlib==3.8.4
- rainbow-api==1.0.9
- pybaselines==1.1.0
- tqdm

## Note on Development and Testing

This package has been developed and tested using the Spyder IDE. While it is expected to work in other development environments, it has not been specifically tested with other IDEs. If you encounter any issues when using the package in a different environment, please feel free to raise an issue or reach out for support.

## Usage

Once installed, you can start using chromatoPy to analyze your HPLC and GC-FID chromatographic data. Below is a basic example demonstrating how to use the package to load and analyze data:

## Example

```python
import chromatopy

# Convert chromatography data to .csv files
chromatopy.hplc_to_csv()

# Run interactive analyzer
chromatopy.hplc_integration()

# Calculate and assign indices
chromatopy.assign_indices()

# Run FID integration
chromatopy.FID_integration()
```

## Input Data Requirements

chromatoPy expects HPLC data to be in **.csv** format. You can convert your raw HPLC results using the hplc_to_csv() function. FID data are assumed to derive from the software Chromeleon (i.e. a .txt file with raw data following the line "Chromatogram Data Information:"


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
