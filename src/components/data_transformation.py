import sys
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:    
    preprocessor_ob_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
            This function is responsible for Data Transformation.
        """
        try:
            num_cols = []   # Numerical Columns
            cat_cols = []   # Categorical Columns            
            
            # Pipeline for Categorical Variables
            num_pipeline = Pipeline(
                steps = [
                   # ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler())                   
                ]) 

            # Pipeline for Categorical Variables
            cat_pipeline = Pipeline(
                steps=[
                    #("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OrdinalEncoder())
                    ("scalar", StandardScaler())
                ]
            )

            # logging.info("Numerical Columns Standard Scaling Completed")
            # logging.info("Numerical Columns Encoding Completed")
            logging.info(f"Numerical Columns   : {num_cols}")
            logging.info(f"Categorical Columns : {cat_cols}")

            preprocessor = ColumnTransformer(       # Joining the Pipelines
                [
                    ("num_pipeline", num_pipeline, num_cols),       # For the Numerical Variables
                    ("cat_pipeline", cat_pipeline, cat_cols)        # For the Categorical Variables
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of Train and Test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "is_fraud"            
            numerical_columns = ["amt"]
            categorical_columns = ["category", "gender", "job", "city", ""]

            """
            =========================================================
            We have 12 numerical features : 
            ['trans_date_trans_time', 'cc_num', 'amt', 
            'zip', 'lat', 'long', 'city_pop', 'dob', 'unix_time', 
            'merch_lat', 'merch_long', 'is_fraud']
            =========================================================
            We have 10 categorical features : 
            ['merchant', 'category', 'first', 'last', 
            'gender', 'street', 'city', 'state', 'job', 
            'trans_num']
            =========================================================
            """

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, 
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, 
                             np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)