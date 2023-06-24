import sys
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:    
    preprocessor_ob_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:

            num_cols = []   # Numerical Columns
            cat_cols = []   # Categorical Columns
            
            # Pipeline for Categorical Variables
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler())                   
                ]) 

            # Pipeline for Categorical Variables
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                    ("scalar", StandardScaler())
                ]
            )

            logging.info("Numerical Columns Standard Scaling Completed")
            logging.info("Numerical Columns Encoding Completed")


            preprocessor = ColumnTransformer(       # Joining the Pipelines
                [
                    ("num_pipeline", num_pipeline, num_cols),       # For the Numerical Variables
                    ("cat_pipeline", cat_pipeline, cat_cols)        # For the Categorical Variables

                ]
            )

            


        except: