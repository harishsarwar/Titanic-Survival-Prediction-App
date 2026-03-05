import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, X):
        try:
            # creating column transformers and preprocessing pipeline

            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])


            logging.info(f"categorical columns : {categorical_features}")
            logging.info(f"numerical columns : {numeric_features}")

            preprocessor = ColumnTransformer([
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])


            return preprocessor

        except Exception as e:
            logging.error(CustomException(e, sys))

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            # Dropping unwanted columns from train data
            train_df.drop(['name', 'ticket','cabin','boat','body','home.dest'], axis=1, inplace=True)

            test_df = pd.read_csv(test_path)
            # Dropping unwanted columns from test data
            test_df.drop(['name', 'ticket','cabin','boat','body','home.dest'], axis=1, inplace=True)


            logging.info("Read train and test data completed")
            
            # Intializing target variable and features
            target_column = 'survived'
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]

            preprocessor_obj = self.get_data_transformer_object(X_train)

            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Concatenating transformed features with target variable column wise

            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            logging.info("Data transformation is completed")

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)
        

        except Exception as e:
            logging.error(CustomException(e, sys))