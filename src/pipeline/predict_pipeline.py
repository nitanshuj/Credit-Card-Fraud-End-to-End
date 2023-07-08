import sys
import os
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", preprocessor.pkl)            
            print("Before Loading")            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)            
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, 
                 category: str, 
                 state: str, 
                 amt: float, 
                 gender: int, 
                 age: int, 
                 city_pop: int,
                 year_transaction:int, 
                 month_transaction:int):
        
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
            pass
        except Exception as e:
            raise CustomException(e,sys)    