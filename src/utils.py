import os
import sys
import dill
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
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
    


def evaluate_and_tune_models(X_train, y_train, X_test, y_test, models):
    param_grids = get_param_grids()

    model_report = {}
    best_model_score = float("-inf")
    best_model_name = None
    best_model = None

    for model_name, model in models.items():
        params = param_grids.get(model_name, {})

        if params:
            grid = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=5,
                scoring="r2",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            tuned_model = grid.best_estimator_
        else:
            tuned_model = model
            tuned_model.fit(X_train, y_train)

        y_pred = tuned_model.predict(X_test)
        score = r2_score(y_test, y_pred)

        model_report[model_name] = score

        if score > best_model_score:
            best_model_score = score
            best_model_name = model_name
            best_model = tuned_model

    return model_report, best_model_name, best_model_score, best_model

def get_param_grids():
    return {
        "Random Forest": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
        },
        "Decision Tree": {
            "max_depth": [None, 5, 10],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1],
        },
        "Linear Regression": {},  # no hyperparameters
        "K-Neighbors Regressor": {
            "n_neighbors": [3, 5, 7],
        },
        "XGBRegressor": {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1],
        },
        "CatBoosting Regressor": {
        }
    }
