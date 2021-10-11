#!/usr/bin/env python
# coding: utf-8

# In[82]:


import streamlit as st
import pandas as pd


# In[83]:


import keras
from keras.models import load_model


# In[84]:


model = load_model('iv_fraudmodel_oct5_bestrecall.h5')


# # Creating Streamlit Application

# In[85]:


st.header('**Image Validation Fraud Detection Application**')
st.subheader('Upload IV Data to get the CIDs predicted as Fraud')


# In[86]:


def excel_file_merge(a,b):
    a.columns = a.columns.str.upper()
    b.columns = b.columns.str.upper()
    c = pd.merge(a, b, on='CUSTOMER_ID', how='inner')
    return c


# In[87]:


def predict_fraud(df):
    prediction_data = df.drop(['COMPLAINT_CREATED_DATE', 'DELIVERY_DATE', 'CUSTOMER_ID', 'COMPLAINT_ID', 'MILK_ORDERS', 'OLD'], axis = 1)
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    scaler = StandardScaler()
    testing_data_scaled = scaler.fit_transform(prediction_data)
    y_pred = model.predict(testing_data_scaled)
    df['predicted_fraud'] = y_pred
    fraud_predictions = df[df['predicted_fraud']>0.5]
    fraud_cid = pd.DataFrame(fraud_predictions['COMPLAINT_ID'].unique())
    return st.download_button(label="Download data as CSV", data=fraud_cid.to_csv().encode('utf-8'),file_name='Fraud_CIDs.csv', mime='text/csv')


# In[88]:


with st.sidebar.header('Upload your Real Time Data'):
    real_time_data = st.sidebar.file_uploader("Real-Time Data", type=["csv"])
    st.sidebar.markdown("""Update every 15 mins""")


# In[89]:


with st.sidebar.header('Upload your Aggregate Data'):
    aggregate_data = st.sidebar.file_uploader("Aggregate Data", type=["csv"])
    st.sidebar.markdown("""Update once a day""")


# In[90]:


# with st.sidebar.header('**File Upload**'):
#     uploaded_file = st.sidebar.file_uploader("Upload your Fraud data in CSV format", type=["csv"])


# In[91]:


if real_time_data is not None:
    real_time_data_csv = pd.read_csv(real_time_data)
    aggregate_data_csv = pd.read_csv(aggregate_data)
    merged_data = excel_file_merge(real_time_data_csv, aggregate_data_csv)
    st.markdown('**Glimpse of dataset**')
    st.write(merged_data)
    predict_fraud(merged_data)


# In[ ]:




