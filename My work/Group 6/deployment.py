# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:25:48 2023

@author: Nishant
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st



st.markdown('''  
<style>
.stApp {
    
    background-image: url('https://img.favpng.com/7/24/2/blood-vessel-heart-circulatory-system-artery-health-png-favpng-S6yh410xF88wYhQPgM5DJxuB7_t.jpg');
    background-color:#white;\
    align:center;\
    display:fill;\
    border-radius: false;\
    border-style: solid;\
    border-color:#white;\
    border-style: false;\
    border-width: 2px;\
    color:black;\
    font-size:14px;\
    font-family: Source Sans Pro;\
    background-color:#white;\
    text-align:center;\
    letter-spacing:0.1px;\
    padding: 0.1em;">\
}

.sidebar {
    background-color: BLUE;
    
}

.  {
        width: 2000px;
        }
.st-b7 {
    color: #White;
}
.css-nlntq9 {
    font-family: Source Sans Pro;
}
</style>
''', unsafe_allow_html=True)



# loading the saved model
loaded_model = pickle.load(open(r'D:\ExcelR\Data Science\Project (Myocardial infarction)\Group 6\rf_model.pkl','rb'))


# creating a function for Prediction

def heartattack_prediction(input_data):
  
    # changing the input_data to numpy array
    input_np = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_np.reshape(1,-1)
    
   # scaler = StandardScaler()
   # scaled_data = scaler.fit_transform(input_data_reshaped)
    
    # Use inverse_transform() to convert the scaled data back to the original format
    #original_data = scaler.inverse_transform(scaled_data)
    
    prediction = loaded_model.predict(input_data_reshaped)
    

    if (prediction == 0):
      return 'Unknown (alive)'
    elif (prediction == 1):
        return 'Cardiogenic shock'
    elif (prediction == 2):
        return 'Pulmonary edema'
    elif (prediction == 3):
        return 'Myocardial Rupture'
    elif (prediction == 4):
        return 'Progress of congestive heart failure'
    elif (prediction == 5):
        return 'Thromboembolism'
    elif (prediction == 6):
        return 'Asystole'
    elif (prediction == 7):
        return 'Ventricular fibrillation'
        
  
    
  
def main():
    

    # giving a title
    st.title('Classification of Myocardial Infarction')

             
    # getting the input data from the user
    st.sidebar.title("Input the Myocardial Infarction Factors:")
    
    Age =  st.sidebar.number_input("Enter an Age:", value=0.0)
            
    Gender = st.sidebar.selectbox("Select the Gender of the person : 0-Female, 1-Male",("0","1"))
        
    RAZRIV = st.sidebar.selectbox("Select the Myocardial Rupture : 0-NO, 1-YES",("0","1"))
        
    D_AD_ORIT = st.sidebar.number_input("Enter the Diastolic blood pressure according to intensive care unit:", value=0.0)
        
    S_AD_ORIT =  st.sidebar.number_input("Enter the Systolic blood pressure according to intensive care unit:", value=0.0)
        
    SIM_GIPERT = st.sidebar.selectbox("Select the Symptomatic Hypertension : 0-NO, 1-YES",("0.0","1.0"))
        
    ROE = st.sidebar.number_input("Enter an ESR (Erythrocyte sedimentation rate):", value=0.0)
        
    K_SH_POST = st.sidebar.selectbox("Select the Cardiogenic shock at the time of admission to intensive care unit: 0-NO, 1-YES",("0.0","1.0"))
        
    TIME_B_S = st.sidebar.selectbox("Select the  Time elapsed from the beginning of the attack of CHD to the hospital : ",("1.0","2.0","3.0","4.0","5.0","6.0","7.0","8.0","9.0"))
       
    R_AB_3_n = st.sidebar.selectbox("Select the Relapse of the pain in the third day of the hospital period : ",("0.0","1.0","2.0","3.0"))
        
    ant_im = st.sidebar.selectbox("Select the Presence of an anterior myocardial infarction : ",("0.0","1.0","2.0","3.0","4.0"))
       
    AST_BLOOD = st.sidebar.number_input("Enter Serum AsAT content:", value=0.0)
        
    IBS_POST = st.sidebar.selectbox("Select the Coronary heart disease (CHD) in recent weeks, days before admission to hospital ",("0.0","1.0","2.0"))
        
    nr07 = st.sidebar.selectbox("Select the Ventricular fibrillation in the anamnesis: 0-NO, 1-YES",("0.0","1.0"))
        
    
        
    # creating a button for Prediction
        
    if st.button('Survival Test Result'):
            result = heartattack_prediction([Age,Gender,RAZRIV,D_AD_ORIT,S_AD_ORIT,SIM_GIPERT,ROE,K_SH_POST,TIME_B_S,R_AB_3_n,ant_im,AST_BLOOD,IBS_POST,nr07])
            st.success('Classification of Myocardial Infarction is {}'.format(result))
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_resume = st.file_uploader("Upload your file Here :", type={"csv"})  

    if uploaded_resume is not None:
        df= pd.read_csv(uploaded_resume,index_col=0)
        #st.write(df)
        input_na = np.asarray(df)
        predict = loaded_model.predict(input_na)
        #st.write(predict)
        # convert the NumPy array to a pandas Series
        series = pd.DataFrame(predict) #, columns='LET_IS')

        concate = pd.concat([df, series],axis = 1)
        st.write(concate)
if __name__ == '__main__':
    main()
    
    
    
    
    
    