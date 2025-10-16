import streamlit as st
import pandas as pd
import joblib
import os

# Load saved components
model = joblib.load("models/model.pkl")
num_imputer = joblib.load("models/num_imputer.pkl")
cat_imputer = joblib.load("models/cat_imputer.pkl")
encoders = joblib.load("models/encoder.pkl")
# Load expected columns
feature_columns = joblib.load('feature_columns.pkl')

# Define columns (must match training)
num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
cat_cols = list(encoders.keys())

st.title("Chronic Kidney Disease (CKD) Prediction")

with st.form("ckd_form"):
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0)
    bp = st.number_input("Blood Pressure (bp)", min_value=0.0, value=80.0)
    sg = st.selectbox("Specific Gravity (sg)", ['1.005', '1.010', '1.015', '1.020', '1.025'])
    al = st.selectbox("Albumin (al)", ['0', '1', '2', '3', '4', '5'])
    su = st.selectbox("Sugar (su)", ['0', '1', '2', '3', '4', '5'])
    rbc = st.selectbox("Red Blood Cells (rbc)", ['normal', 'abnormal'])
    pc = st.selectbox("Pus Cell (pc)", ['normal', 'abnormal'])
    pcc = st.selectbox("Pus Cell Clumps (pcc)", ['present', 'notpresent'])
    ba = st.selectbox("Bacteria (ba)", ['present', 'notpresent'])
    bgr = st.number_input("Blood Glucose Random (bgr)", min_value=0.0, value=120.0)
    bu = st.number_input("Blood Urea (bu)", min_value=0.0, value=35.0)
    sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, value=1.2)
    sod = st.number_input("Sodium (sod)", min_value=0.0, value=135.0)
    pot = st.number_input("Potassium (pot)", min_value=0.0, value=4.5)
    hemo = st.number_input("Hemoglobin (hemo)", min_value=0.0, value=15.0)
    pcv = st.number_input("Packed Cell Volume (pcv)", min_value=0.0, value=42.0)
    wc = st.number_input("White Blood Cell Count (wc)", min_value=0.0, value=7500.0)
    rc = st.number_input("Red Blood Cell Count (rc)", min_value=0.0, value=5.2)
    htn = st.selectbox("Hypertension (htn)", ['yes', 'no'])
    dm = st.selectbox("Diabetes Mellitus (dm)", ['yes', 'no'])
    cad = st.selectbox("Coronary Artery Disease (cad)", ['yes', 'no'])
    appet = st.selectbox("Appetite", ['good', 'poor'])
    pe = st.selectbox("Pedal Edema (pe)", ['yes', 'no'])
    ane = st.selectbox("Anemia (ane)", ['yes', 'no'])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input data
    input_dict = {
        'age': [age],
        'bp': [bp],
        'sg': [float(sg)],
        'al': [float(al)],
        'su': [float(su)],
        'rbc': [rbc],
        'pc': [pc],
        'pcc': [pcc],
        'ba': [ba],
        'bgr': [bgr],
        'bu': [bu],
        'sc': [sc],
        'sod': [sod],
        'pot': [pot],
        'hemo': [hemo],
        'pcv': [pcv],
        'wc': [wc],
        'rc': [rc],
        'htn': [htn],
        'dm': [dm],
        'cad': [cad],
        'appet': [appet],
        'pe': [pe],
        'ane': [ane]
    }
    input_df = pd.DataFrame(input_dict)

    # Impute missing values
    input_df[num_cols] = num_imputer.transform(input_df[num_cols])
    input_df[cat_cols] = cat_imputer.transform(input_df[cat_cols])

    # Encode categorical columns
    for col in cat_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    # Ensure correct column order for prediction
    input_df = input_df.reindex(columns=feature_columns)

    prediction = model.predict(input_df)[0]
    prediction_label = "CKD" if prediction == 1 else "Not CKD"

    st.subheader("🔍 Prediction Result:")
    st.success(f"The patient is likely: **{prediction_label}**")
