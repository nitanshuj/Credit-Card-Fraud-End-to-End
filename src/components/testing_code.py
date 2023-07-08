
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer




if __name__=="__main__":
    
    print("Data Ingestion Started!!")
    
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    print("Data Ingestion Complete!!\n")

    print("Data Transformation Started!!")
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    
    print("Data Transformation Completed!!\n")

    modeltrainer=ModelTrainer()
    
    print("Training Started!!")
    
    modeltrainer.initiate_model_trainer(train_arr,test_arr)
    
    print("Training Completed!!")
