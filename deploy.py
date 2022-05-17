# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:43:44 2022

@author: intanh
"""

import pickle
import os
import numpy as np
import streamlit as st

# Constant
SCALER_PATH = os.path.join(os.getcwd(),'scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), "model.pkl")

# Data Loading
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
    
# Model
with open(MODEL_PATH, 'rb') as g:
    scaler = pickle.load(g)
    
chance = {0:"LOW", 1:"HIGH"}

patience_info = np.array([55,1,0,132,353,0,1,132,1,1.2,1,1,3])
patience_info = scaler.transform(np.expand_dims(patience_info, axis=0))

new_pred = scaler.predict(patience_info)
if np.argmax(new_pred) == 0:
   new_pred = [0,1]
   print(chance[np.argmax(new_pred)])
else:
   new_pred = [1,0]
   print(chance[np.argmax(new_pred)])
    
#%% Streamlit

with st.form('Heart Attack Prediction Form'):
    st.write("Patient's Info")
    age = int(st.number_input("Age:")) # add int because not float
    sex = int(st.number_input("Sex") )
    cp = int(st.number_input("Chest Pain type:"))
    trtbps = st.number_input("Resting blood pressure (in mm Hg):") # not need int because of float value
    chol = st.number_input("Cholestoral (in mg/dl):")
    fbs = int(st.number_input("Fasting blood sugar:"))
    restecg = int(st.number_input("Resting electrocardiographic:"))
    thalachh = st.number_input("Maximum heart rate:")
    exng = int(st.number_input("Exercise induced angina:"))
    oldpeak = st.number_input("Previous Peak:")
    slp = int(st.number_input("Slope:"))
    caa = int(st.number_input("Number of major vessels (0-3):"))
    thall = int(st.number_input("Thal rate:"))
    
    submitted = st.form_submit_button('Submit')
    
    if submitted == True:
        patience_info = np.array([age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall])
        patience_info = scaler.transform(np.expand_dims(patience_info, axis=0))
        new_pred = scaler.predict(patience_info)
        if np.argmax(new_pred) == 1:
            st.warning
            (f"Possibility: {chance[np.argmax(new_pred)]}")
        else:
            st.snow()
            st.success
            (f"Possibility: {chance[np.argmax(new_pred)]}")