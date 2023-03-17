import os
import re, warnings
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame, Series
from typing import Tuple
from numpy import ndarray
import seaborn as sns
from matplotlib import pyplot as plt

"""
COLOR SETTING

%matplotlib inline

THEME_STYLE = "darkgrid"
THEME_PALETT = "pastel"

sns.set_theme(palette="pastel")
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

https://seaborn.pydata.org/generated/seaborn.set_theme.html
https://matplotlib.org/stable/tutorials/colors/colormaps.html
https://stackoverflow.com/questions/48958208/how-do-you-change-the-default-font-color-for-all-text-in-matplotlib/48958263#48958263

THEME_PALETT = "Dark2"
THEME_STYLE = "ticks"# "darkgrid"
BG_COLOR= "black" # "darkslategray" # "midnightblue"# "dimgray" #"black"
# BG_COLOR="grey"
TEXT_COLOR= "snow" #"lightgrey"
sns.set_theme(style=THEME_STYLE, palette=THEME_PALETT)
sns.set(rc={'axes.facecolor': BG_COLOR, 'figure.facecolor':BG_COLOR,
            'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR, 'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR })
"""

"""
https://stackoverflow.com/questions/25238442/setting-plot-background-colour-in-seaborn

import seaborn
theme_style = "white"
seaborn.set_theme(style=theme_style)
# bg_color="back"
# seaborn.set(rc={'axes.facecolor': bg_color, 'figure.facecolor':bg_color})


matplotlab dark color https://gist.github.com/mwaskom/7be0963cc57f6c89f7b2

temporay styling: https://matplotlib.org/stable/tutorials/introductory/customizing.html#temporary-styling
"""


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

def assign_const_col(df: DataFrame, col_name: str, value: any) -> DataFrame:
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html
    # https://stackoverflow.com/questions/29517072/add-column-to-dataframe-with-constant-value
    return df.assign(col_name=value)

@dataclass
class KaggleData:
    train_path: str = ""
    test_path: str = ""
    label_col: str = ""

    def __post_init__(self):
        self.train_X = pd.DataFrame()
        self.train_y = pd.Series(dtype= float)
        self.test_X = pd.DataFrame()
    
    def _cache_empty(self):
        """ Test if all cached DataFrame and Series are empty"""
        # testing all the cached dataframe and series shall not be empty
        # https://stackoverflow.com/questions/42360956/what-is-the-most-pythonic-way-to-check-if-multiple-variables-are-not-none
        return None not in map(lambda x: x.empty, (self.train_X, self.train_y, self.test_X) )
    
    def _train_test_X(self):
        return pd.concat([self.train_X, self.test_X], ignore_index=True)

    def _load(self) -> Tuple[DataFrame, DataFrame]:
        try:
            train_X_y_df = pd.read_csv(self.train_path, sep=",", header=0)
            test_X_df = pd.read_csv(self.test_path, sep=",", header=0)
        except: 
            train_X_y_df = pd.DataFrame()
            test_X_df = pd.DataFrame() 
        return train_X_y_df, test_X_df
    
    def load(self) -> Tuple[DataFrame, DataFrame, Series]:
        train_X_y_df, test_X_df = self._load()
        # train_X_y_df and test_X_df may have different size of column,
        # thus the mask must be calculated separately
        train_column_mask: ndarray = ~train_X_y_df.columns.isin([self.label_col])
        test_column_mask: ndarray = ~test_X_df.columns.isin([self.label_col])
        # cache to this object
        self.train_X = train_X_y_df.loc[:, train_column_mask]
        self.train_y = train_X_y_df[self.label_col]
        self.test_X = test_X_df.loc[:, test_column_mask]
        return self.train_X, self.test_X, self.train_y
    
    def load_all(self) -> DataFrame:
        # testing all the cached dataframe and series shall not be empty
        if self._cache_empty():
            _, _, _ = self.load()
        return self._train_test_X()
    
    def _all_data_sets(self) -> DataFrame:
        if self._cache_empty():
            _, _, _ = self.load()
        all_dist_df = pd.DataFrame()    
        for df, location in zip(
            (self.train_X, self.test_X, self._train_test_X()),
            ("train", "test", "total")
        ):  
            # add the data_partition attribute to all data sample 
            all_dist_df = pd.concat([all_dist_df, df.assign(data_partition=location)])
        return all_dist_df

    def boxplot_dist(self, features=[], orient="v", marker="x", legend="lower right", dark_mode=True):
        """
        @param legend: 
            upper right
            upper left
            lower left
            lower right
            right
            center left
            center right
            lower center
            upper center
            center
        @param edgecolor: matplotlib color, css colors https://matplotlib.org/stable/gallery/color/named_colors.html    
        """
        if dark_mode:
            edge_color = "lightgray" # lightgray, snow # for box
            # https://matplotlib.org/stable/tutorials/colors/colormaps.html
            # seaborn matplotlib color palett: https://seaborn.pydata.org/tutorial/color_palettes.html
            palett = "bright" # deep, muted, pastel, bright, dark, and colorblind
            text_color= "snow" #"lightgrey"
            bg_color= "black"
            # theme_palett = "Dark2"
            # theme_style = "darkgrid" # "ticks"
            # bg_color= "black" # "darkslategray" # "midnightblue"# "dimgray" #"black", "grey"
            # text_color= "snow" #"lightgrey"
            # sns.set_theme(style=theme_style, palette=theme_palett)
            # sns.set(rc={'axes.facecolor': bg_color, 'figure.facecolor':bg_color,
            #             'text.color': text_color, 'axes.labelcolor': text_color, 
            #             'xtick.color': text_color, 'ytick.color': text_color })
        else:
            edge_color = "black"
            text_color= "black"
            bg_color= "white"
            palett =  "pastel" # bright        

        all_df = self._all_data_sets()
        all_num_cols = all_df.describe().columns.to_list() + ["data_partition"]
        if features is None or len(features)==0:
            cols = all_num_cols
        else:     
            cols_set = set(all_df.describe().columns.to_list())
            features_set = set(features)
            common_set = cols_set.intersection(features_set)
            common_set.add("data_partition")
            if len(common_set) == 1: 
                # only has the data_partition
                cols = all_num_cols
                warnings.warn(f"numerical features {features} are not found, displays all numerical features", category=UserWarning, stacklevel=1)
            else:    
                cols = list(common_set)
        
        # https://stackoverflow.com/questions/42004381/box-plot-of-a-many-pandas-dataframes/42005692#42005692
        # the pd.melt create the value and features column, where feature column encode all the feature   
        mdf = pd.melt(all_df[cols], id_vars=['data_partition'], var_name=["numerical_features"])
        """boxplot all numerical features for the different datasets 'train', 'test', 'total' """

        # make the edge of boxes white
        # https://stackoverflow.com/questions/43434020/black-and-white-boxplots-in-seaborn/65529178#65529178
        # 
        PROPS = {
            #'boxprops':{'facecolor':'none', 'edgecolor':'red'},
            'boxprops':{'edgecolor':edge_color},
            'medianprops':{'color':edge_color},
            'whiskerprops':{'color':edge_color},
            'capprops':{'color':edge_color},
            'flierprops' :{"marker" : marker, "markerfacecolor":edge_color, "markeredgecolor": edge_color},
            'palette': palett
        }

        def local_plot(orient: str, df:DataFrame, PROPS):
            match orient:
                case "h": 
                    # ax = sns.boxplot(y="data_partition", x="value", hue="numerical_features", data=mdf, flierprops={"marker": marker}, **PROPS)
                    ax = sns.boxplot(y="data_partition", x="value", hue="numerical_features", data=df, **PROPS)
                case "v", _:
                    # ax = sns.boxplot(x="data_partition", y="value", hue="numerical_features", data=mdf, flierprops={"marker": marker, }, **PROPS)
                    ax = sns.boxplot(x="data_partition", y="value", hue="numerical_features", data=df, **PROPS)


        
        if dark_mode:
            with plt.style.context('dark_background'):
                sns.set(rc={'axes.facecolor': bg_color, 'figure.facecolor':bg_color,
                    'text.color': text_color, 'axes.labelcolor': text_color, 
                    'xtick.color': text_color, 'ytick.color': text_color })
                local_plot(orient=orient, df=mdf, PROPS=PROPS)
        else:     
            local_plot(orient=orient, df=mdf, PROPS=PROPS)
        plt.legend(loc=legend)
        plt.show()        
            
           


