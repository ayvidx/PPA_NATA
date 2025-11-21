import streamlit as st
import pandas as pd
import joblib, json, os

st.set_page_config(page_title='🛒 Customer Analytics App', layout='centered')
st.title('🛒 Customer Analytics App')
st.write('Prediction (Random Forest) + Classification (KMeans)')

MODEL_RF = 'rf_model.pkl'
MODEL_KMEANS = 'kmeans_model.pkl'
FEATURE_FILE = 'feature_columns.json'
CLUSTER_FILE = 'cluster_labels.json'
INTERPRET_FILE = 'interpretation.txt'

# Load models
rf_model, kmeans_model, feature_cols, cluster_labels, interpretation_text = None, None, None, None, None

if os.path.exists(MODEL_RF):
    rf_model = joblib.load(MODEL_RF)
    st.success('✅ RF Model loaded successfully')
else:
    st.error('❌ RF Model file not found')

if os.path.exists(MODEL_KMEANS):
    kmeans_model = joblib.load(MODEL_KMEANS)
    st.success('✅ KMeans Model loaded successfully')
else:
    st.error('❌ KMeans Model file not found')

if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, 'r') as f:
        feature_cols = json.load(f)
else:
    st.error('❌ Feature list not found')

if os.path.exists(CLUSTER_FILE):
    with open(CLUSTER_FILE, 'r') as f:
        cluster_labels = json.load(f)
else:
    st.error('❌ Cluster labels not found')

if os.path.exists(INTERPRET_FILE):
    with open(INTERPRET_FILE, 'r') as f:
        interpretation_text = f.read()

# Input form
st.header('Enter Customer Details:')
inputs = {}
for col in feature_cols:
    inputs[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([inputs])
st.subheader('Input Preview')
st.dataframe(input_df)

# Prediction
if st.button('🔮 Predict Spending'):
    if rf_model is None or feature_cols is None:
        st.error('Model or feature list not loaded.')
    else:
        X_pred = input_df[feature_cols]
        pred = rf_model.predict(X_pred)[0]
        st.success(f'💰 Predicted Spending: {pred:.2f}')

# Classification
if st.button('🧩 Classify Segment'):
    if kmeans_model is None or feature_cols is None:
        st.error('Model or feature list not loaded.')
    else:
        X_pred = input_df[feature_cols]
        cluster = kmeans_model.predict(X_pred)[0]
        st.success(f'📊 Assigned Cluster: {cluster} → {cluster_labels[str(cluster)]}')
        st.text(interpretation_text)
