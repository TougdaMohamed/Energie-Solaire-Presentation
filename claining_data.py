import pandas as pd

# load your data into a DataFrame
df = pd.read_csv('2021-11-jqro_hora_L1.csv')

# drop rows that contain any null value
df = df.dropna()

# drop columns that contain any null value
df = df.dropna(axis=1)

# save the cleaned data to a new CSV file
df.to_csv('cleaned_2021-11-jqro_hora_L1.csv', index=False)
