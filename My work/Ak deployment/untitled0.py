# -*- coding: utf-8 -*-
"""
Created on Tue May 16 02:15:10 2023

@author: HP
"""

import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
import pickle

# Load the pickle files for the models
random_forest_model = pickle.load(open('rf_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
xgboost_model = pickle.load(open('xgb_model.pkl', 'rb'))

# Function to perform data preprocessing
def preprocess_data(df):
    # Get columns with more than 30% missing values
    missing_percentages = df.isna().mean() * 100
    cols_with_more_than_30percent_missing = missing_percentages[missing_percentages > 30].index

    # Drop columns with more than 30% missing values
    df.drop(columns=cols_with_more_than_30percent_missing, inplace=True)

    # Drop rows with more than 30% missing values
    row_list = []
    for i in range(df.shape[0]):
        n_miss = df.iloc[i].isnull().sum()
        perc = n_miss / df.shape[1] * 100
        if perc >= 30:
            row_list.append(i)
    df.drop(row_list, inplace=True)

    # Create an instance of SimpleImputer with strategy='most_frequent'
    imputer = SimpleImputer(strategy='most_frequent')
    # Fill missing values with the mode
    x_filled = imputer.fit_transform(df)
    # Convert the filled array back to a DataFrame
    df = pd.DataFrame(x_filled, columns=df.columns)

    return df

# Function to make predictions using the models
def predict(model, data):
    # Perform any necessary preprocessing on the data
    
    # Make predictions
    predictions = model.predict(data)
    
    return predictions

# Streamlit UI
def main():
    st.title('Model Prediction')
    
    # Add file upload option for external data
    uploaded_file = st.file_uploader('Upload external data', type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file as DataFrame
            df = pd.read_csv(uploaded_file)
            
            # Preprocess the data
            preprocessed_df = preprocess_data(df)
            
            # Apply SMOTE to the preprocessed data
            # Perform feature scaling on the resampled data
            # Apply PCA for dimensional reduction
            # Apply Lasso regularization
            # Filter the PCA components based on non-zero coefficients from Lasso regularization
            # Prepare the final DataFrame for predictions
            
            # Make predictions using the models
            rf_predictions = predict(random_forest_model, preprocessed_df)
            svm_predictions = predict(svm_model, preprocessed_df)
            xgb_predictions = predict(xgboost_model, preprocessed_df)
            
            # Display the predictions
            st.subheader('Random Forest Predictions')
            st.write(rf_predictions)
            
            st.subheader('SVM Predictions')
            st.write(svm_predictions)
            
            st.subheader('XGBoost Predictions')
            st.write(xgb_predictions)
            
        except Exception as e:
            st.error(f'Error: {e}')

if __name__ == '__main__':
    main()