import os
import sys
import dill
from src.exception import CustomException


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
    
