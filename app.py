import streamlit as st
import pandas as pd
import joblib
import os
import io
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report

# 1. Page Configuration
st.set_page_config(page_title="BITS ML Assignment 2", layout="wide")
st.title("🍷 Wine Quality Classifier (BITS Assignment 2)")

# 2. Sidebar Settings
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox(
    "Select Model",
    ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"]
)

# 3. File Upload Logic
st.write("### 1. Upload Test Data")
uploaded_file = st.file_uploader("Upload Test CSV (Wine Quality)", type=["csv"])

if uploaded_file is not None:
    # --- FIX: Read content once to avoid EmptyDataError ---
    content = uploaded_file.getvalue().decode('utf-8')
    
    # Try semicolon separator first (Common for this dataset)
    df = pd.read_csv(io.StringIO(content), sep=';')
    
    # If the file didn't split (only 1 column), try comma
    if len(df.columns) <= 1:
        df = pd.read_csv(io.StringIO(content), sep=',')
    
    st.write("### 2. Dataset Preview")
    st.dataframe(df.head())

    # 4. Load the pre-trained model
    model_path = f"model/{model_option}.pkl"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # 5. Feature Engineering
        # Drop quality/target so we only have the 11 chemical features
        X_test = df.drop(columns=['quality', 'target'], errors='ignore')
        
        try:
            # 6. Prediction
            y_pred = model.predict(X_test)
            
            # 7. Evaluation
            if 'quality' in df.columns:
                # Convert quality to binary (7+ is Good/1, rest is 0)
                y_true = (df['quality'] >= 7).astype(int)
                
                acc = accuracy_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                
                st.write(f"### 3. Performance: {model_option}")
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
                st.info("Upload a file with a 'quality' column to see performance metrics.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Check if your CSV has exactly the same features as the training data.")
    else:
        st.error(f"File '{model_option}.pkl' not found in model/ folder.")

else:
    st.info("Awaiting CSV upload. Please use the sidebar or upload area.")
