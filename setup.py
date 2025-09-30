from setuptools import setup, find_packages

setup(
    name="hecras-tools",
    version="0.1.0",
    description="HEC-RAS geometry and data operations utilities",
    author="Kushal Regmi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "geopandas",
        "shapely",
        "rasterio",
        "h5py",
        "joblib"
    ],
)
