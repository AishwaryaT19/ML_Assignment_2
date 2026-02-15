import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

st.title("Wine Quality Classifier (BITS Assignment 2)")

# a. Dataset upload [cite: 91]
uploaded_file = st.file_uploader("Upload Test CSV (Wine Quality)", type="csv")

# b. Model selection [cite: 92]
model_name = st.selectbox("Select Model", ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file, sep=';')
    # Pre-processing if quality column exists
    if 'quality' in test_df.columns:
        y_true = (test_df['quality'] >= 7).astype(int)
        X_test = test_df.drop('quality', axis=1)
    else:
        X_test = test_df

    # Load selected model
    model = joblib.load(f"model/{model_name}.pkl")
    preds = model.predict(X_test)

    # c. Display metrics [cite: 93]
    st.subheader(f"Performance: {model_name}")
    
    # d. Confusion Matrix [cite: 94]
    if 'quality' in test_df.columns:
        st.text("Classification Report:")
        st.text(classification_report(y_true, preds))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_true, preds))
    else:
        st.write("Predictions for uploaded data:")
        st.write(preds)