import pandas as pd
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv("data/kidney_disease_labeled.csv")  # your actual file path

# Drop ID column
df.drop(columns=['id'], inplace=True)

# Strip whitespace from string values
df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

# Fix label issues
df['classification'] = df['classification'].replace({'ckd\t': 'ckd'})

# Convert object columns with numeric values to float
for col in ['pcv', 'wc', 'rc']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing values
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()

num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Save preprocessed data
df.to_csv("data/1kidney_disease_preprocessed.csv", index=False)
