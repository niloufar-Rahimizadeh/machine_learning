# In typical environments your data would be available 
# in a relational database (or some other common data 
# store) and spread across multiple tables/documents/files.
import os
import tarfile
import urllib.request


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
print(HOUSING_PATH)
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# when you call fetch_housing_data(), it creates a datasets/housing directory in
# your workspace, downloads the housing.tgz file, and extracts the housing.csv 
# file from it in this directory.


def fetch_housing_data(housing_url=HOUSING_URL, housing_path= HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz") #dataset/housing/housing.tgz
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()