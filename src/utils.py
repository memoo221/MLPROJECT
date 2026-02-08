import os
import sys
import dill
from src.exception import CustomException

from sklearn.metrics import r2_score

import os
import pickle

def save_object(file_path: str, obj):
    """
    Saves a Python object to a pickle file.

    Parameters:
    - file_path (str): Path where the pickle file will be saved
    - obj: Any Python object to be pickled
    """
    try:# Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Save object as pickle
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models):
    model_report = {}

    best_model_score = float("-inf")
    best_model_name = None
    best_model = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = r2_score(y_test, y_pred)
        model_report[model_name] = score

        if score > best_model_score:
            best_model_score = score
            best_model_name = model_name
            best_model = model

    return model_report, best_model_name, best_model_score, best_model

