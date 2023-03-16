import os
import re
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame
from typing import Tuple

def current_dir():
    """get the absolute path of the current file directory"""
    current_path = os.path.dirname(os.path.dirname(__file__))
    return current_path

def current_dir_subpath(subpath: str):
    """
    @param subpath: "data/train.csv", no leading "/"
    """
    # replace the leading / with ""
    return os.path.join(current_dir(), re.sub(r"^\/+" , "" , subpath)) 

@dataclass
class KaggleData:
    train_path: str = ""
    test_path: str = ""
            
    def load(self) -> Tuple[DataFrame, DataFrame]:
        try:
            train_df = pd.read_csv(self.train_path, sep=",", header=0)
            test_df = pd.read_csv(self.test_path, sep=",", header=0)
        except: 
            train_df = pd.DataFrame()
            test_df = pd.DataFrame() 
        return train_df, test_df
    
    def load_all(self) -> DataFrame:
        train_df, test_df = self.load()
        return pd.concat([train_df, test_df], ignore_index=True) 