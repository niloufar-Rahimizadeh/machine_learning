import pandas as pd 
import os
import matplotlib.pyplot as plt


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
print(HOUSING_PATH)
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
print(housing.head())
housing.info()


print(housing["ocean_proximity"].value_counts())
print(housing.describe())
housing.hist(bins=50, figsize=(20,15))
plt.show()