import sys
import os
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", 'preprocessor.pkl')            
            print("Before Loading")            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            # print("data scaled ran!!")
            preds = model.predict(data_scaled)            
            # print("preds generated")
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, category, state, 
                 amt, gender, age, 
                 city_pop, year_transaction, 
                 month_transaction):
        
        self.category = category
        self.state = state
        self.amt = amt
        self.gender = gender
        self.age = age
        self.city_pop = city_pop
        self.year_transaction = year_transaction
        self.month_transaction = month_transaction

    def get_data_as_data_frame(self):
        try:
            custom_Data_input_dict = {
                "category":[self.category],
                "state":[self.state],
                "amt":[self.amt],
                "gender":[self.gender],
                "age":[self.age],
                "city_pop":[self.city_pop],
                "year_transaction":[self.year_transaction],
                "month_transaction":[self.month_transaction]
            }
            return pd.DataFrame(custom_Data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)    
        

