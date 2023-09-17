import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self) -> None:
        self.data_ingestor = DataIngestion()
        self.data_transformer = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self) -> float:
        try:
            logging.info('Initiating training pipeline ...')
            train_path, test_path = self.data_ingestor.initiate_data_ingestion()
            train_array, test_array, _ = self.data_transformer.initiate_data_transformation(
                train_path=train_path, test_path=test_path
            )
            best_r2_score = self.model_trainer.initiate_model_training(train_array=train_array, test_array=test_array)

            return best_r2_score

        except Exception as e:
            logging.info('Exception occurred at training pipeline stage.')
            raise CustomException(e, sys)


if __name__ == '__main__':
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()
