import os
import sys
from dataclasses import dataclass

# Importing Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Importing other utilities
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve, matthews_corrcoef

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]                
            )
        
            models = {
                "Logistic_Regression": LogisticRegression(),
                "Random_Forest": RandomForestClassifier(),
                # "Support Vector Machine": SVC(),
                # "Naive Bayes": GaussianNB()            
            }

            params = {
                "Logistic_Regression":{
                    'penalty':['l1', 'l2'],
                    'class_weight':['balanced']
                },
                "Random_Forest":{
                    'criterion':['squared_error'],
                    'class_weight':['balanced'],
                    'n_estimators':[8, 16]
                }
            }

            logging.info("Model Training Begins")
            
            print("Model Training Begins!")

            model_report:dict = evaluate_models(X_train=X_train, 
                                                y_train=y_train, 
                                                X_test=X_test, 
                                                y_test=y_test, 
                                                models=models)  #X_train, y_train, X_test, y_test, models, param

            model_report = evaluate_models(X_train=X_train,
                                           y_train=y_train, 
                                           X_test=X_test, 
                                           y_test=y_test, 
                                           models=models, 
                                           param=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model                
            )
            
            predicted=best_model.predict(X_test)

            recall = recall_score(y_test, predicted)
            print("Recall Score: ", recall)

            return recall

        except Exception as e:
            raise CustomException(e, sys)