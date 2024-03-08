# -*- coding: utf-8 -*-
"""
Created on Tue May 16 00:07:49 2023

@author: HP
"""

import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
import pickle
from sklearn.impute import SimpleImputer

# Load saved models
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('svm_model.pkl', 'rb') as g:
    svm_model = pickle.load(g)

with open('xgb_model.pkl', 'rb') as h:
    xgb_model = pickle.load(h)

# Define functions for preprocessing and predicting on user uploaded data

def preprocess_data(df):
    
    # Handle missing values
    missing_percentages = df.isna().mean() * 100
    cols_with_more_than_30percent_missing = missing_percentages[missing_percentages > 30].index
    df.drop(columns=cols_with_more_than_30percent_missing, inplace=True)
    
    columns_to_drop = ['FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    row_list = []
    for i in range(df.shape[0]):
        n_miss = df.iloc[i].isnull().sum()
        perc = n_miss / df.shape[1] * 100
        if perc >= 30:
            row_list.append(i)

    df.drop(row_list, inplace=True)
    
    # Create an instance of SimpleImputer with strategy='most_frequent'
    imputer = SimpleImputer(strategy='most_frequent')
    
    # Fill missing values in the DataFrame with the mode
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Apply SMOTE to the data
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    x_resampled, y_resampled = smote.fit_resample(df.drop('LET_IS', axis=1), df['LET_IS'])
    
    # Perform feature scaling on the resampled data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_resampled)
    
    # Apply PCA for dimensional reduction on the scaled data
    pca = PCA(n_components=0.95, random_state=42)
    x_pca = pca.fit_transform(x_scaled)
    
    # Create a dataframe of PCA results for the data
    pca_columns = [f'pc{i+1}' for i in range(x_pca.shape[1])]
    pca_df = pd.DataFrame(x_pca, columns=pca_columns)
    
    # Filter the PCA components based on non-zero coefficients from Lasso regularization
    lasso = Lasso(alpha=0.1)  # Adjust the regularization parameter alpha as per your needs
    lasso.fit(pca_df, y_resampled)
    selected_features = lasso.coef_ != 0
    pca_df_selected = pca_df.loc[:, selected_features]
    
    return pca_df_selected

def predict_on_data(df):
    
    # Preprocess the data
    pca_df = preprocess_data(df)
    
    # Make predictions using the saved models
    rf_preds = rf_model.predict(pca_df)
    svm_preds = svm_model.predict(pca_df)
    xgb_preds = xgb_model.predict(pca_df)
    
    # Combine the predictions into a single dataframe
    preds_df = pd.DataFrame({'Random Forest': rf_preds, 'SVM': svm_preds, 'XGBoost': xgb_preds})
    
    return preds_df

# Define the Streamlit app
st.title('Classification of Myocardial Infarction')
st.write('Upload a CSV file to make predictions on')

uploaded_file = st.file_uploader('Choose a file')


if uploaded_file is not None:
    
    # Read in the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Make predictions
    preds_df = predict_on_data(df)
    
    # Show the predictions in a table
    st.write('Predictions:')
    st.write(preds_df)