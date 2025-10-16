import pandas as pd

# Load the dataset
df = pd.read_csv("data/kidney_disease.csv")  # adjust path as needed

# Display the first few rows
print("First 5 rows of dataset:")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Class distribution
print("\nTarget Variable Distribution:")
print(df['classification'].value_counts())
