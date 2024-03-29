import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def get_data_transformer_obj():
        try:
            numerical_col = ['reading_score', 'writing_score']
            categorical_col = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                               'test_preparation_course']

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Numerical columns: {numerical_col}')
            logging.info(f'Categorical columns: {categorical_col}')

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_col),
                    ('categorical_pipeline', categorical_pipeline, categorical_col)
                ]
            )

            logging.info('Preprocessor is ready!')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Data Transformation started ...')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Done reading train and test data.')
            logging.info('Obtaining preprocessor object.')

            preprocessing_obj = self.get_data_transformer_obj()
            target_col_name = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Saved preprocessing object.')

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.info('Exception occurred in Data Transformation.')
            raise CustomException(e, sys)
