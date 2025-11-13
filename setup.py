from setuptools import setup, find_packages

setup(
    name="mpc_ship_nav",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "shapely",
        "pyproj",
        "matplotlib",
        "fiona",
    ]
)
