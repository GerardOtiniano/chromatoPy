from setuptools import setup, find_packages

setup(
    name="chromatopy",
    version="0.1.2",
    description="An open-source package for integrating HPLC chromatography data using a flexible multigaussian and single gaussian fitting algorithm. This package requires the user to convert raw HPLC results using openChrom, and tjem converting the openChrom results to .csv files.",
    packages=find_packages(),
    install_requires=[
        # list libraries your package depends on
        "numpy==1.26.4",  # for example
        "pandas==2.2.2",
        "scikit-learn==1.4.2",
        "scipy==1.13.1",
        "setuptools==69.5.1",
        "matplotlib==3.8.4",
        "rainbow==2.8.0",
    ],
    python_requires=">=3.12.4",
    author="Gerard Otiniano",
    author_email="gerardot@buffalo,edu",
)
