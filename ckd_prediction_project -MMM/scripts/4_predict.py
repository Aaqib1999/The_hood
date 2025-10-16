import pandas as pd
import numpy as np
import joblib

# Load saved components
model = joblib.load("models/model.pkl")
num_imputer = joblib.load("models/num_imputer.pkl")
cat_imputer = joblib.load("models/cat_imputer.pkl")
encoders = joblib.load("models/encoder.pkl")

cat_cols = list(encoders.keys())
num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

def preprocess_input(input_df):
    # Impute missing values
    input_df[num_cols] = num_imputer.transform(input_df[num_cols])
    input_df[cat_cols] = cat_imputer.transform(input_df[cat_cols])

    # Encode categorical columns
    for col in cat_cols:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])

    return input_df

# Example input (can modify this for testing or live input)
new_data = {
    'age': [65],
    'bp': [90],
    'sg': ['1.010'],
    'al': ['3'],
    'su': ['1'],
    'rbc': ['abnormal'],
    'pc': ['abnormal'],
    'pcc': ['present'],
    'ba': ['present'],
    'bgr': [150],
    'bu': [40],           # <-- Add this line (example value)
    'sc': [5.2],
    'sod': [130],
    'pot': [4.2],
    'hemo': [9.0],
    'pcv': [28],
    'wc': [9800],
    'rc': [3.9],
    'htn': ['yes'],
    'dm': ['yes'],
    'cad': ['no'],
    'appet': ['poor'],
    'pe': ['yes'],
    'ane': ['yes']
}

# Create DataFrame and predict
input_df = pd.DataFrame(new_data)
processed = preprocess_input(input_df)
prediction = model.predict(processed)

print("Prediction:", "CKD" if prediction[0] == 1 else "Not CKD")
