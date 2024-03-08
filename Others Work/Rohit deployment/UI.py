import numpy as np
import pickle
import streamlit as st
import pandas as pd

page_bg_img = '''
<style>
.stApp {
background-image: url("https://wallpaper.dog/large/10866895.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Loading model
with open(r"D:\ExcelR\Data Science\Project (Myocardial infarction)\Rohit Deployment\Classification.pkl", 'rb') as f:
    models = pickle.load(f)

# Access individual models
model1 = models['model1']
model2 = models['model2']
model3 = models['model3']


def main():
    
    title = '<p style="font-family:sans-serif; color:Blue; font-size: 45px; text-align:center; ">Myocardinal Infraction Test</p>'
    st.markdown(title, unsafe_allow_html=True)
    
    subtitle = '<p style="font-family:sans-serif; color:yellow; font-size: 25px; ">Enter data to check your health condition.</p>'
    st.markdown(subtitle, unsafe_allow_html=True)
    
    subtitle1 = '<p style="font-family:sans-serif; color:orange; font-size: 20px; ">Give your medical information in CSV format and get report on your health.</p>'
    st.markdown(subtitle1, unsafe_allow_html=True)
    
    subtitle2 = '<p style="font-family:sans-serif; color:green; font-size: 15px; ">Your CSV file should have data as follows :</p>'
    st.markdown(subtitle2, unsafe_allow_html=True)
    df = pd.read_csv(r'D:\ExcelR\Data Science\Project (Myocardial infarction)\Rohit Deployment\Input_file_format.csv')
    
    @st.cache_data
    def convert_df(df): 
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)
    
    # Downloading CSV file
    st.download_button(
        label="Download CSV file format :arrow_down:",
        data=csv,
        file_name='Input file format.csv',
        mime='csv',)
    
    subtitle3 = '<p style="font-family:sans-serif; color:green; font-size: 15px; ">Here is information about the parameters in CSV file :</p>'
    st.markdown(subtitle3, unsafe_allow_html=True)
    
    with open(r"D:\ExcelR\Data Science\Project (Myocardial infarction)\Rohit Deployment\Input_file_info.txt", 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Downloading TXT file
    st.download_button(
        label="Download information file :arrow_down:",
        data=content,
        file_name='Input file info.txt',
        mime='txt',)
    
    uploaded_file = st.file_uploader('Choose CSV file :')
    
    if uploaded_file is not None:
    
        selection = st.selectbox('What type of model do you want?', (None, 'Random Forest Classifier',
                                                                        "eXtreme Gradient Boosting", "Artificial Neural Network"))
        
        a = 'Random Forest Classifier'
        b = "eXtreme Gradient Boosting"
        c = "Artificial Neural Network"
        
        if selection == None:
            warn = """<p style="font-family:sans-serif; color:Red; font-size: 15px; ">You haven't selected your preference for model to be used.</p>"""
            st.markdown(warn, unsafe_allow_html=True)
        
        if selection == a:
            st.write('You have selected Random Forest Classifier model')
            st.write(' ')
            
            
            Input = pd.read_csv(uploaded_file, index_col=0)
            st.write('Your uploaded file is :')
            st.write(Input)
            
            if st.button('Classify'):
                classification = model2.predict(Input)
                Input['Outcome'] = classification
                st.write('Classification is as follows :')
                st.write(classification)
                st.write('Dataframe with classification :')
                Input
                
                @st.cache_data
                def convert_df(df): 
                    return df.to_csv().encode('utf-8')

                csv = convert_df(Input)

                st.download_button(
                    label="Download data as CSV file :arrow_down:",
                    data=csv,
                    file_name='Report.csv',
                    mime='csv',)
        
        if selection == b:
            st.write('You have selected eXtreme Gradient Boosting Classifier model')
            st.write(' ')
            
            
            Input = pd.read_csv(uploaded_file, index_col=0)
            st.write('Your uploaded file is :')
            st.write(Input)
            
            if st.button('Classify'):
                classification = model1.predict(Input)
                Input['Outcome'] = classification
                st.write('Classification is as follows :')
                st.write(classification)
                st.write('Dataframe with classification :')
                Input
                
                @st.cache_data
                def convert_df(df): 
                    return df.to_csv().encode('utf-8')

                csv = convert_df(Input)

                st.download_button(
                    label="Download data as CSV file :arrow_down:",
                    data=csv,
                    file_name='Report.csv',
                    mime='csv',)
                
        if selection == c:
            st.write('You have selected Artificial Neural Network Classifier model')
            st.write(' ')
            
            
            Input = pd.read_csv(uploaded_file, index_col=0)
            st.write('Your uploaded file is :')
            st.write(Input)
            
            if st.button('Classify'):
                classification = model3.predict(Input)
                classification = np.argmax(classification, axis=-1)
                Input['Outcome'] = classification
                st.write('Classification is as follows :')
                st.write(classification)
                st.write('Dataframe with classification :')
                Input
                
                @st.cache_data
                def convert_df(df): 
                    return df.to_csv().encode('utf-8')

                csv = convert_df(Input)

                st.download_button(
                    label="Download data as CSV file :arrow_down:",
                    data=csv,
                    file_name='Report.csv',
                    mime='csv',)
        
    else:
        warn = """<p style="font-family:sans-serif; color:Red; font-size: 15px; ">You have not uploaded CSV file</p>"""
        st.markdown(warn, unsafe_allow_html=True)
    
    st.write('')
    st.write('') 
    st.write('') 
    st.write('')
    
    if st.button('About Us'):
        warn = """<p style="font-family:sans-serif; color:black; font-size: 15px; ">Group 2 of project P219.</p>"""
        st.markdown(warn, unsafe_allow_html=True)
    
    if st.button('Contact'):
        mail = 'group2-p219@excelr.com'
        st.write('E-mail :', mail)
        warn = """<p style="font-family:sans-serif; color:black; font-size: 15px; ">Number = '+91-xxxxxxxxxx'</p>"""
        st.markdown(warn, unsafe_allow_html=True)    



if __name__ == '__main__':
    main()