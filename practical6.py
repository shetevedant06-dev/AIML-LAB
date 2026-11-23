import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

dataset_url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/adult/adult.data"
)
feature_names = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "income"
]
raw_df = pd.read_csv(
    dataset_url,
    names=feature_names,
    skipinitialspace=True,
    na_values="?"
)

print("First five records:")
print(raw_df.head())
print("\nDataset shape:", raw_df.shape)

print("\nMissing values before processing:")
print(raw_df.isnull().sum())

num_columns = raw_df.select_dtypes(include=["int64", "float64"]).columns
for col in num_columns:
    raw_df[col] = raw_df[col].fillna(raw_df[col].median())

cat_columns = raw_df.select_dtypes(include=["object", "category"]).columns
for col in cat_columns:
    raw_df[col] = raw_df[col].fillna(raw_df[col].mode()[0])

print("\nMissing values after processing:")
print(raw_df.isnull().sum())

processed_df = pd.get_dummies(
    raw_df,
    columns=cat_columns.drop("income").tolist(),
    dtype=int
)
processed_df["income"] = LabelEncoder().fit_transform(processed_df["income"])

print("\nDataset after encoding:")
print(processed_df.head())

scale_cols = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss",
    "hours_per_week"
]
minmax = MinMaxScaler()
processed_df[scale_cols] = minmax.fit_transform(processed_df[scale_cols])

print("\nDataset after normalization:")
print(processed_df.head())

processed_df.to_csv("census_income_cleaned.csv", index=False)
print("\nProcessed dataset saved successfully!")
print("Final dataset shape:", processed_df.shape)