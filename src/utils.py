import logging
import os
import sys
import dill
from rich.progress import track
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        logging.info('Evaluating models ...')
        logging.info(f'All models: {models.items()}')

        report = {}
        for i in track(range(len(models)), description="Doing Science..."):
            model = list(models.values())[i]
            print(model)
            logging.info(f'Model: {model}')

            para = param[list(models.keys())[i]]
            print('Grid Search parameters: ', para)
            logging.info(f'Model parameters: {para}')

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(x_train, y_train)
            logging.info(f'Grid search completed!')

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)
            print(f'Best parameters obtained from GridSearch: {gs.best_params_}')
            logging.info(f'Best parameters obtained from GridSearch: {gs.best_params_}')

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            print('Train r2_score: ', train_model_score)
            logging.info(f'Train r2_score: {train_model_score}')

            test_model_score = r2_score(y_test, y_test_pred)
            print('Test r2_score: ', test_model_score)
            logging.info(f'Test r2_score: {test_model_score}')

            report[list(models.keys())[i]] = test_model_score
            print('x-----------------------------------------------------------------------x')

        logging.info('All models evaluated, and report generated with best performance on test data.')

        return report

    except Exception as e:
        logging.info('Error occurred while evaluating models!')
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        logging.info(f'Loading object on path: {file_path}')
        with open(file_path, 'rb') as file_object:
            return dill.load(file_object)

    except Exception as e:
        logging.info(f'Error occurred while loading object on path: {file_path}')
        raise CustomException(e, sys)
