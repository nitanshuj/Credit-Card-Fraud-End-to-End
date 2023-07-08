from flask import Flask,request,render_template
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            category=request.form.get('category'),
            state=request.form.get('state'),
            amt=request.form.get('amt'),
            gender=request.form.get('gender'),
            age=request.form.get('age'),
            city_pop=request.form.get('city_pop'),
            year_transaction=request.form.get('year_transaction'),
            month_transaction=request.form.get('month_transaction')
            )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        
        

        if int(results[0]) == 0:    
            res = "Not Fraud"
        else: 
            res = "Fraud"   

        return render_template('home.html',results=res)
        

if __name__=="__main__":
    app.run(host="0.0.0.0")