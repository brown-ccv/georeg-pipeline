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
- Install tesseract: brew update && brew upgrade && brew install tesseract
- Install tesserocr: conda install -c derickl tesserocr
- Install weighted-levenshtein: pip install weighted-levenshtein

- Within ~/anaconda3/envs/georeg/etc/conda/activate.d (might be anaconda2), create an env_vars.sh script that sets the passwords for the Brown ArcGIS geocoder.  You may have to ask for this file.
- Within ~/anaconda3/envs/georeg/etc/conda/deactivate.d (might be anaconda2), create an env_vars.sh script that unsets the passwords for the Brown ArcGIS geocoder.# georeg-pipeline
A pipeline for extracting data from City Directories

### For adding dependencies

1) Install Python 2.7.
2) Clone git repository
3) Type 'pip install -r requirements.txt' to install dependencies.


