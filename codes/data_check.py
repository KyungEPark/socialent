import pandas as pd

data = pd.read_parquet('socialent/data/output/transparency_pred_Qwen3-8B_pred_reason.parquet')
print(data.head())
print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.isnull().sum())
print(data.describe())
print(data.info())


data.to_csv('socialent/data/output/transparency_pred_Qwen3-8B_pred_reason.csv', index=False)