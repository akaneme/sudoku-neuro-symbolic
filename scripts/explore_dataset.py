# here i confirmed what actually was in da dataset

import pandas as pd 

df = pd.read_csv("data/raw/sudoku-3m.csv")

print(df.head())
print()

print(df.columns)
print()

print(df.info())
print()

print(df.describe())
print()

print(df.describe())

print(df.isnull().sum())
print(df.iloc[0])
