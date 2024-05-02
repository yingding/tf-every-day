import pandas as pd
import os

print(os.getcwd())
df = pd.read_csv('06PandasPlots/proband.csv')
print(df)


def has_same_gender(group_df: pd.DataFrame):
    """
    check the length of the unique values for gender in the grouped studyid
    == 1 if all the values are the same, >= 1 if there are different values
    """
    return len(group_df['gender'].unique()) <= 1

def has_different_gender(group_df: pd.DataFrame):
    """
    check the length of the unique values for gender in the grouped studyid
    == 1 if all the values are the same, >= 1 if there are different values
    """
    return len(group_df['gender'].unique()) > 1

df_dif_gender = df.groupby(['studyid']).filter(has_different_gender)
print(df_dif_gender)
# print(type(df_dif_gender))

# remove the rows in df_dif_gender from df
# https://stackoverflow.com/questions/44706485/how-to-remove-rows-in-a-pandas-dataframe-if-the-same-row-exists-in-another-dataf/44706892#44706892
def remove_raws(left_df, right_df):
    """
    remove raws in the left_df that are in the right_df
    """
    return pd.merge(left_df, right_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
# df has the same gender in the group is the original df minus the rows with different gender
df_has_same_gender = remove_raws(df, df_dif_gender)
print(df_has_same_gender)