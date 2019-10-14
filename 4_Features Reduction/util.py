from termcolor import colored
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

############### Show colored text #############


def bg(value, type='num', color='blue'):
    value = str('{:,}'.format(value)) if type == 'num' else str(value)
    return colored(' ' + value + ' ', color, attrs=['reverse', 'blink'])


############ Print the variable name ##############
# Credits: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
import inspect


def var2str(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def shape(*args):
    max_len = 0
    for df in args:
        max_len = max(len(var2str(df)), max_len)
    for df in args:
        print(f'~> [{var2str(df).ljust(max_len)}] has {bg(df.shape[0])} rows, and {bg(df.shape[1])} columns.')


############### Summary Table #####################
from scipy import stats

# Summary dataframe


def summary(df, sort_col=0):
    summary = pd.DataFrame({'dtypes': df.dtypes}).reset_index()
    summary.columns = ['Name', 'dtypes']
    summary['Missing'] = df.isnull().sum().values
    summary['M_Percent'] = round(100 * summary['Missing'] / df.shape[0], 2)
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2), 2)

    summary = summary.sort_values(by=[sort_col], ascending=False) if sort_col else summary

    # Print some smmaries.
    print(f'~> Dataframe has {bg(df.shape[0])} Rows, and {bg(df.shape[1])} Columns.')
    print(f'~> Dataframe has {bg(summary[summary["Missing"] > 0].shape[0], color="red")} Columns have [Missing] Values.')
    print('---' * 20)
    for type_name in np.unique(df.dtypes):
        print(f'~> There are {bg(df.select_dtypes(type_name).shape[1])}\t Columns that have [Type] = {bg(type_name, "s", "green")}')

    return summary.style.background_gradient(cmap='summer_r')


from tqdm import tqdm_notebook


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**3
    print('~> Memory usage of dataframe is {:.3f} GB'.format(start_mem))

    for col in tqdm_notebook(df.columns):
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # Comment this if you have NaN value in this column.
        # else:
            # df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 3
    print('~> Memory usage after optimization is: {:.3f} GB'.format(end_mem))
    print('~> Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    print('---' * 20)
    return df
