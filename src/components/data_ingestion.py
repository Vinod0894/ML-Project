import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_tranformation import DataTransfromationConfig
from src.components.data_tranformation import DataTransformtion
from src.components.model_trainer import ModelTrainner
from src.components.model_trainer import ModelTrainnerConfig

@dataclass
class DataIngestionConfig():
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_conifg=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as Dataframe")
            os.makedirs(os.path.dirname(self.ingestion_conifg.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_conifg.raw_data_path,index=False,header=True)

            logging.info ("Train test split Initiated")
            train_set,test_set =train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_conifg.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_conifg.test_data_path,index=False,header=True)
            logging.info("Ingestion of Data is completed")
            return(
                self.ingestion_conifg.train_data_path,
                self.ingestion_conifg.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_tranformation =DataTransformtion()
    train_arr,test_arr,_ = data_tranformation.initiate_data_tranformation(train_data,test_data)

    model_trainner=ModelTrainner()
    print(model_trainner.initiate_model_trainner(train_arr,test_arr))