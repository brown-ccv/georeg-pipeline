![Static Badge](https://img.shields.io/badge/ARCHIVED-red)

**This repo was archived February 2026, due to inactivity.**

A pipeline for extracting data from City Directories

This is a Python 2 code, so bear that in mind during installation

How to install necessary packages:
- Install the georeg anaconda environment: conda create -n georeg -c brown-data-science tesserocr
- Load georeg anaconda environment: source activate georeg (this may be conda activate georeg for you)
- You may have to unset the python path if you get an error message: unset PYTHONPATH
- Install opencv3: conda install -c menpo opencv3
- Install pandas: conda install pandas
- Install zipcode: pip install zipcode
- Install sqlalchemy: conda install sqlalchemy
- Install geopy: conda install -c conda-forge geopy
- Install PIL: conda install PIL
- Install python-levenshtein: conda install -c conda-forge python-levenshtein
- Install tesseract: brew update && brew upgrade && brew install tesseract
- Install tesserocr: conda install -c mcs07 tesserocr
- Install matplotlib: conda install matplotlib
- Install sklearn: pip install sklearn
- Install fuzzywuzzy: pip install fuzzywuzzy

- Within ~/anaconda3/envs/georeg/etc/conda/activate.d (might be anaconda2), create an env_vars.sh script that sets the passwords for the Brown ArcGIS geocoder.  You may have to ask for this file.
- Within ~/anaconda3/envs/georeg/etc/conda/deactivate.d (might be anaconda2), create an env_vars.sh script that unsets the passwords for the Brown ArcGIS geocoder.# georeg-pipeline

You will need to produce a StreetZipCity.csv file for your area.  It can be missing the zipcode data, which are not necessary to the code.


