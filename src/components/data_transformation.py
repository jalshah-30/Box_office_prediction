import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from scipy.sparse import hstack


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_transformer_object(self):
        '''This function is responsible for data transformation'''
        try:
            categorical = [
                "original_language",
                "genres",
                "cast",
                "directors",
                # "production_companies"
            ]

            numerical = [
                "vote_count",
                # "popularity",
                # "director_avg_boxoffice",
                # "actor_avg_boxoffice",
                "budget"
            ]
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        Pipeline([
                            ("imputer", SimpleImputer(strategy="median"))
                        ]),
                        numerical
                    ),
                    (
                        "cat",
                        Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))
                        ]),
                        categorical
                    )
                ]
            )
            logging.info("Encoding & Imputing Completed")
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)

    def initiate_Data_Transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_transformer_object()
            target_column_name="collection"
            numerical_column = [
                            "vote_count",
                            # "popularity",
                            # "director_avg_boxoffice",
                            # "actor_avg_boxoffice",
                            "budget"
                        ]

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = hstack([
                input_feature_train_arr,
                np.array(target_feature_train_df).reshape(-1, 1)
            ])

            test_arr = hstack([
                input_feature_test_arr,
                np.array(target_feature_test_df).reshape(-1, 1)
            ])

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
            

    
