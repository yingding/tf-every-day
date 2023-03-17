import os
import re
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame, Series
from typing import Tuple
from numpy import ndarray

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
            
    def _load(self) -> Tuple[DataFrame, DataFrame]:
        try:
            train_X_y_df = pd.read_csv(self.train_path, sep=",", header=0)
            test_X_df = pd.read_csv(self.test_path, sep=",", header=0)
        except: 
            train_X_y_df = pd.DataFrame()
            test_X_df = pd.DataFrame() 
        return train_X_y_df, test_X_df
    
    def load(self, label_col="Survived") -> Tuple[DataFrame, DataFrame, Series]:
        train_X_y_df, test_X_df = self._load()
        # train_X_y_df and test_X_df may have different size of column,
        # thus the mask must be calculated separately
        train_column_mask: ndarray = ~train_X_y_df.columns.isin([label_col])
        test_column_mask: ndarray = ~test_X_df.columns.isin([label_col])
        return train_X_y_df.loc[:, train_column_mask], test_X_df.loc[:, test_column_mask], train_X_y_df[label_col]
    
    def load_all(self, label_col="Survived") -> DataFrame:
        train_X_df, test_X_df, _ = self.load(label_col)
        return pd.concat([train_X_df, test_X_df], ignore_index=True) 