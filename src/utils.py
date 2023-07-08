import os
import sys
import dill
import pickle

import numpy as np 
import pandas as pd

from datetime import date, datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# ==============================================================================

def feature_engineering(train_df, test_df):
    """
        Feature Engineering
    """        
    
    # Get the current date & converting it into datetime
    current_date = datetime.combine(date.today(), datetime.min.time())

    # Working on train data
    train_df["trans_date_trans_time"] = pd.to_datetime(train_df["trans_date_trans_time"], format='ISO8601')
    train_df["dob"] = pd.to_datetime(train_df["dob"], format='ISO8601') 
    train_df['age'] = (current_date - train_df['dob']//pd.Timedelta(days=365.25))
    train_df['year_transaction'] = train_df['trans_date_trans_time'].dt.year
    train_df['month_transaction'] = train_df['trans_date_trans_time'].dt.month
    train_df['gender'] = train_df['gender'].apply(lambda x: 0 if x == 'F' else 1)
    train_df['year_transaction'] = train_df['year_transaction'].apply(lambda x: 0 if x==2019 else 1)
    train_df = softcapping(train_df, 'amt', 0.1, 0.99) # Softcapping for Outlier Treatment
    train_df.set_index('trans_num', inplace=True)

    # Working on test data
    test_df["trans_date_trans_time"] = pd.to_datetime(test_df["trans_date_trans_time"], format='ISO8601')
    test_df["dob"] = pd.to_datetime(test_df["dob"], format='ISO8601') 
    test_df['age'] = (current_date - test_df['dob']//pd.Timedelta(days=365.25))
    test_df['year_transaction'] = test_df['trans_date_trans_time'].dt.year
    test_df['month_transaction'] = test_df['trans_date_trans_time'].dt.month    
    test_df['gender'] = test_df['gender'].apply(lambda x: 0 if x == 'F' else 1)
    test_df['year_transaction'] = test_df['year_transaction'].apply(lambda x: 0 if x==2019 else 1)
    test_df.set_index('trans_num', inplace=True)
    
    return train_df, test_df

# ==============================================================================

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# ==============================================================================

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# ==============================================================================

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# ==============================================================================

def show_null_percentage(df):
    
    """
    Function to Show details of the Percentage of NULL values
        df : dataframe for which NULL value percentage have to be displayed
    """
        
    count_missing_values = df.isnull().sum()
    percent_missing_values = df.isnull().sum()/ len(df)
    missing_percent_df = pd.DataFrame({'column_name': df.columns, 
                                       'number_of_missing_values':count_missing_values,
                                       'percent_missing': percent_missing_values})
    
    missing_percent_df.drop(['column_name'], axis=1, inplace=True)    
    missing_percent_df.sort_values('percent_missing', inplace=True, ascending = False)
    missing_percent_df['percent_missing'] = round(missing_percent_df['percent_missing']*100, 2)
    result_df = missing_percent_df[missing_percent_df['percent_missing'] > 0]
    return result_df

# ==============================================================================

def softcapping(df, col,lower_percentile, higher_percentile):
    """
    Function for Soft Capping Outliers
        This function soft caps the values or removes values lower 
        than a certain percentile and higher than a certain percentile.
    """
    percentiles = df[col].quantile([lower_percentile, higher_percentile]).values
    if lower_percentile > 0:
        df = df[df[col]>=percentiles[0]]
    if higher_percentile < 100:
        df = df[df[col]<=percentiles[1]]
    return df

# ==============================================================================

