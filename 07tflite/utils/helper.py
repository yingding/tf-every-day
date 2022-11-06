from __future__ import annotations
# data_exchange_path = create_default_data_exchange_path(exchange_file_name="previous_training_results_dataclass")
import os
import pickle
from dataclasses import dataclass, field

"""
Google Style multi-line doc string
https://www.datacamp.com/tutorial/docstrings-python
"""


@dataclass
class PreviousTrainingResults():
    """
    param: epochs using field(default_factory=list) method instead of [] to avoid utable default not allowed error
    https://stackoverflow.com/questions/53632152/why-cant-dataclasses-have-mutable-defaults-in-their-class-attributes-declaratio
    """
    epochs: list = field(default_factory=list)
    losses: list = field(default_factory=list)

    def dump(self, path: str):
        # write a wb, binary model, no encoding argument encoding='UTF-8'
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
    
    @staticmethod
    def load(path: str) -> PreviousTrainingResults:
        with open(path, 'rb') as fp:
            return pickle.load(fp, encoding='UTF-8')   

default_tf_model_dir_path = "model/tf"
default_tflite_model_dir_path = "model/tflite"

default_tf_checkpoint_dir_path = f"{default_tf_model_dir_path}/check_points"
default_tflite_checkpoint_dir_path = f"{default_tflite_model_dir_path}/check_points"

default_tf_savedmodel_dir_path = f"{default_tf_model_dir_path}/saved_model"
default_data_passing_dir_path = "data/share"

def create_default_data_exchange_subfolder() -> str:
    return create_subfolder(default_data_passing_dir_path)

def create_default_tf_checkpoint_subfolder() -> str:
    return create_subfolder(default_tf_checkpoint_dir_path)

def create_default_tflite_checkpoint_path(check_point_name: str) -> str:
    return os.path.join(create_default_tflite_checkpoint_path(), check_point_name)    

def create_default_tflite_checkpoint_subfolder() -> str:
    return create_subfolder(default_tflite_checkpoint_dir_path)   

def create_default_tf_savedmodle_subfolder() -> str:
    return create_subfolder(default_tf_savedmodel_dir_path)

def create_default_tflite_model_path(tflite_file_name: str) -> str:
    return os.path.join(create_subfolder(default_tflite_model_dir_path), tflite_file_name)

def create_default_data_exchange_path(exchange_file_name: str) -> str:
    return os.path.join(create_default_data_exchange_subfolder(), exchange_file_name)    

def create_subfolder(subfolder: str) -> str:
    '''
    Create a relative subfolder of the working directory, with the given 
    subfolder name.
    
    Args:
        subfolder (str): the subfolder name

    Return:
        str: absolute path of the subfolder, or None if not created    
    '''
    # Save the trained weights to a checkpoint
    ## os.getcwd() is the python process working directory, tf-every-day root
    ## current_path = os.path.abspath(os.getcwd())
    
    ## __file__ returns the current helper.py path, since it is in the utils, you need to get the parent dir using dirname twice
    ## https://stackoverflow.com/questions/30218802/get-parent-of-current-directory-from-python-script/30218825#30218825
    current_path = os.path.dirname(os.path.dirname(__file__))
    current_model_path = os.path.join(current_path, subfolder)
    
    if not os.path.exists(current_model_path):
        os.makedirs(current_model_path)
            
    return current_model_path
    
     
       