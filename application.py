import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Title for the application
st.title("Fraud Detection Application")

# Sidebar for user input
st.sidebar.header("Enter Transaction Details")

category = st.sidebar.text_input("Category")
state = st.sidebar.text_input("State")
amt = st.sidebar.number_input("Transaction Amount", min_value=0.0, format="%f")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=0, format="%d")
city_pop = st.sidebar.number_input("City Population", min_value=0, format="%d")
year_transaction = st.sidebar.number_input("Year of Transaction", min_value=2000, max_value=2100, format="%d")
month_transaction = st.sidebar.number_input("Month of Transaction", min_value=1, max_value=12, format="%d")

# Predict button
if st.sidebar.button("Predict"):
    # Create custom data object
    data = CustomData(
        category=category,
        state=state,
        amt=amt,
        gender=gender,
        age=age,
        city_pop=city_pop,
        year_transaction=year_transaction,
        month_transaction=month_transaction
    )

    # Convert input to DataFrame
    pred_df = data.get_data_as_data_frame()
    st.write("Input Data:", pred_df)

    # Load prediction pipeline
    predict_pipeline = PredictPipeline()

    # Make prediction
    results = predict_pipeline.predict(pred_df)

    # Display results
    if int(results[0]) == 0:
        st.success("Prediction: Not Fraud")
    else:
        st.error("Prediction: Fraud")





# from flask import Flask,request,render_template
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# application = Flask(__name__)
# app = application

# # Route for a home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predictdata', methods=['GET', 'POST'])
# def predict_datapoint():
#     if request.method == 'GET':
#         return render_template('home.html')
#     else:
#         data = CustomData(
#             category=request.form.get('category'),
#             state=request.form.get('state'),
#             amt=request.form.get('amt'),
#             gender=request.form.get('gender'),
#             age=request.form.get('age'),
#             city_pop=request.form.get('city_pop'),
#             year_transaction=request.form.get('year_transaction'),
#             month_transaction=request.form.get('month_transaction')
#             )

#         pred_df = data.get_data_as_data_frame()
#         print(pred_df)
#         print("Before Prediction")

#         predict_pipeline=PredictPipeline()
#         print("Mid Prediction")

#         results=predict_pipeline.predict(pred_df)
#         print("after Prediction")        
        
#         if int(results[0]) == 0:    
#             res = "Not Fraud"
#         else: 
#             res = "Fraud"   

#         return render_template('home.html',results=res)
        

# if __name__=="__main__":
#     app.run(host="0.0.0.0")