import pandas as pd

data = pd.read_parquet('data/subsettext100_socialco_20250718.parquet')
print(data.head())
print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.isnull().sum())
print(data.describe())
print(data.info())