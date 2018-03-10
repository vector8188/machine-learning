import os
import tarfile
from six.moves import urllib
import pandas as pd


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "/datasets/housing/housing.csv"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    csv_path = os.path.join(housing_path, "housing.csv")
    urllib.request.urlretrieve(housing_url, csv_path)
    # housing_tgz = tarfile.open(tgz_path)
    # housing_tgz.extractall(path=housing_path)
    # housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

    
