from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    data_ingestor = DataIngestion()
    train_data_path, test_data_path = data_ingestor.initiate_data_ingestion()
    print('train_data_path: ', train_data_path, 'test_data_path: ', test_data_path)
    data_transformer = DataTransformation()
    train_arr, test_arr, preprocessor_obj_path = data_transformer.initiate_data_transformation(
        train_path=train_data_path, test_path=test_data_path
    )
    model_trainer = ModelTrainer()
    best_r2_score = model_trainer.initiate_model_training(train_array=train_arr, test_array=test_arr)
