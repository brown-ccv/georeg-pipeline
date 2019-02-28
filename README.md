This is a pipeline for extracting data from city directories

### How to install packages

This is tested on Python 3.5.3 and should work on 3.6 as well. First start by setting up the Python virtualenv folder:

``` 
    python3 -m venv /path/to/env
```

Then activate the environment by running:

```
    source /path/to/env/bin/activate
```

Then run the following command to install the packages from pip

```
    pip install -r requirements_py3.txt
```

You will need to install two specific packages, libtesseract and libleptonica, due to the tesserocr package. These are system specific and instructions will vary depending on your OS or distro. Please follow instructions here and Google as necessary: https://github.com/sirfz/tesserocr. 

Once finished, you can deactivate your environment by typing: `deactivate`. 

### Setting up the Brown Geocoder (instructions deprecated)

#### New instructions (to be modified)

Once the env_vars.sh script has been gotten, run it when you activate the environment. That should work at the moment. We are in the process of shifting to an open source geocoder which will change the install instructions. 

#### Older instructions

Within ~/anaconda3/envs/georeg/etc/conda/activate.d (might be anaconda2), create an env_vars.sh script that sets the passwords for the Brown ArcGIS geocoder.  You may have to ask for this file.

Within ~/anaconda3/envs/georeg/etc/conda/deactivate.d (might be anaconda2), create an env_vars.sh script that unsets the passwords for the Brown ArcGIS geocoder.# georeg-pipeline

### Miscellaneous 

You will need to produce a StreetZipCity.csv file for your area.  It can be missing the zipcode data, which are not necessary to the code.

### For MacOS

Build tesserocr this way:

```CC=clang XCC=clang++ CPPFLAGS="-stdlib=libc++ -DUSE_STD_NAMESPACE -mmacosx-version-min=10.8" pip install tesseroc```


