import os
import helper
"""
Google Style multi-line doc string
https://www.datacamp.com/tutorial/docstrings-python
"""
default_checkpoint_dir_path = "model/tmp"
default_savedmodel_dir_path = "model/saved_model"

def create_default_checkpoint_subfolder() -> str:
    return create_subfolder(default_checkpoint_dir_path)

def create_default_savedmodle_subfolder() -> str:
    return create_subfolder(default_savedmodel_dir_path)

def create_default_tflite_model_path(tflite_file_name: str) -> str:
    return os.path.join(create_subfolder("model"), tflite_file_name)

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
    current_path = os.path.dirname(helper.__file__)
    current_model_path = os.path.join(current_path, subfolder)
    
    if not os.path.exists(current_model_path):
        os.makedirs(current_model_path)
            
    return current_model_path
    
     
       