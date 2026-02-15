import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report

# Set page config for a professional look
st.set_page_config(page_title="BITS ML Assignment 2", layout="wide")

st.title("🍷 Wine Quality Classifier (BITS Assignment 2)")

# 1. Sidebar for Model Selection
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox(
    "Select Model",
    ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"]
)

# 2. File Uploader
st.write("### 1. Upload Test Data")
uploaded_file = st.file_uploader("Upload Test CSV (Wine Quality)", type=["csv"])

if uploaded_file is not None:
    # Handle both semicolon and comma delimiters
    df = pd.read_csv(uploaded_file, sep=';')
    if 'quality' not in df.columns:
        df = pd.read_csv(uploaded_file, sep=',')
    
    st.write("### 2. Dataset Preview")
    st.dataframe(df.head())

    # 3. Load the pre-trained model
    model_path = f"model/{model_option}.pkl"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # 4. Prepare Features (X)
        # We must drop 'quality' and 'target' to match the 11 features used in training
        X_test = df.drop(columns=['quality', 'target'], errors='ignore')
        
        # 5. Perform Prediction
        try:
            y_pred = model.predict(X_test)
            
            # 6. Evaluation (If quality column is present)
            if 'quality' in df.columns:
                # Convert quality to binary target (7 or above is 'Good')
                y_true = (df['quality'] >= 7).astype(int)
                
                acc = accuracy_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                
                st.write("### 3. Model Performance")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{acc:.2f}")
                col2.metric("MCC Score", f"{mcc:.2f}")
                
                st.write("#### Confusion Matrix")
                st.write(confusion_matrix(y_true, y_pred))
                
                st.write("#### Classification Report")
                st.text(classification_report(y_true, y_pred))
            else:
                st.write("### 3. Predictions")
                st.write(y_pred)
                st.info("Note: Upload a CSV with a 'quality' column to see accuracy metrics.")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Ensure your CSV has the same 11 chemical features used during training.")
    else:
        st.error(f"Model file '{model_option}.pkl' not found in the 'model/' folder.")

# Instructions for the user
else:
    st.info("Please upload a CSV file to get started.")
