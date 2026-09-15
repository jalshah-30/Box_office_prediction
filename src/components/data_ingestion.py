import os
import sys
from src.exception import CustomException
from src.logger import logging 
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("Notebooks/data/indian_movies.csv")
            df=df[:1808]
            genre_mapping = {
                "Political Drama": "Political",
                "Buddy Comedy": "Comedy",
                "Docudrama": "Documentary",
                "Psychological Drama": "Psychological",
                "Conspiracy Thriller": "Thriller",
                "One-Person Army Action": "Action",
                "Gangster": "Crime",
                "Medical Drama": "Drama",
                "Suspense Mystery": "Suspense",
                "Gun Fu": "Action",
                "Romantic Comedy":"Romantic",
                "Body Horror":"Horror",
                "Psychological Thriller":"Thriller",
                "Quirky Comedy":"Comedy",
                "True Crime":"Crime",
                "Feel-Good Romance":"Romance",
                "Period Drama":"History",
                "Cop Drama":"Cop",
                "Romantic Epic":"Romantic",
                "Legal Thriller":"Legal",
                "Supernatural Horror":"Horror",
                "Artificial Intelligence":"Tech",
                "Desert Adventure":"Adventure",
                "Dark Romance":"Romance",
                "Historical Epic":"History",
                "Political Thriller":"Political",
                "Monster Horror":"Horror",
                "Tragic Romance":"Romance",
                "Psychological Horror":"Horror",
                "Drug Crime":"Crime",
                "Dark Comedy":"Comedy",
                "Showbiz Drama":"Drama",
                "Satire":"Comedy",
                "Legal Drama":"Legal",
                "Jungle Adventure":"Adventure",
                "Police Procedural":"Cop",
                "Teen Romance":"Romance",
                "History Documentary":"History",
                "Sports Documentary":"Sports",
                "Computer Animation":"Anime",
                "Raunchy Comedy":"Comedy",
                "Buddy Cop":"Cop",
                "Swashbuckler":"Adventure",
                "Mockumentary":"Documentary",
                "Cyber Thriller":"Tech",
                "Globetrotting Adventure":"Adventure",
                "Screwball Comedy":"Comedy",
                "Whodunnit":"Crime",
                "Extreme Sport":"Sport",
                "Teen Comedy":"Comedy",
                "Erotic Thriller":"Thriller"
            }

            df["genres"] = df["genres"].replace(genre_mapping)
            df.dropna(inplace=True)

            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the Data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )


        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_Data_Transformation(train_data,test_data)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
