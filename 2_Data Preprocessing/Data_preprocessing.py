'''
Author:      Mosaab Muhammed.
Date:        10/10/2019
Title:       Data Preprocessing.
Description: The aim of this script, is to solve the data related problems,
             such as converting categorical (Nominal & Ordinal) features into numerical,
             and reduce the size of the dataframe for faster read and write.
'''
##################################
#      1.1 Import Libraries:     #
##################################
from termcolor import colored
print(f'################ {colored("1. Importing Libraries", "green")} ################')
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#######################################
#    1.2 Some Utility Functions:      #
#######################################
############### Show colored text #############


def bg(value, type='num', color='blue'):
    value = str('{:,}'.format(value)) if type == 'num' else str(value)
    return colored(' ' + value + ' ', color, attrs=['reverse', 'blink'])


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('~> Memory usage of dataframe is {:.3f} MB'.format(start_mem))

    for col in tqdm(df.columns):
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('~> Memory usage after optimization is: {:.3f} MB'.format(end_mem))
    print('~> Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    print('---' * 20)
    return df


##################################
#       2. Read Data             #
##################################
print(f'################ {colored("2. Reading Data", "green")} #############')
train = pd.read_csv('../1_Data Invistigation & EDA/output/train_null_removed.csv')
test = pd.read_csv('../1_Data Invistigation & EDA/output/test_null_removed.csv')

#################################################
#       3. Categorical ~> Numerical             #
#################################################
print(f'################ {colored("3. Categorical ~> Numerical", "green")} #############')
ord_cols = ['LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtExposure',
            'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'HeatingQC', 'Electrical',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'KitchenQual', 'TotRmsAbvGrd', 'Functional',
            'Fireplaces', 'FireplaceQu', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive',
            'PoolQC', 'BedroomAbvGr', 'KitchenAbvGr']
cat_cols = ['MSSubClass', 'MSZoning', 'Alley', 'LotShape', 'LandContour', 'LotConfig',
            'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'GarageType', 'MiscFeature', 'SaleType',
            'SaleCondition', 'HouseStyle', 'Fence', 'CentralAir']
# 1. Ordinal Features
for dataset in [train, test]:
    dataset.LandSlope = dataset.LandSlope.map({'Sev': 1, 'Mod': 2, 'Gtl': 3})
    dataset.PoolQC = dataset.PoolQC.map({'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
    dataset.PavedDrive = dataset.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})
    dataset.GarageFinish = dataset.GarageFinish.map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
    dataset.KitchenQual = dataset.KitchenQual.map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    dataset.BsmtQual = dataset.BsmtQual.map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    dataset.BsmtCond = dataset.BsmtCond.map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    dataset.ExterQual = dataset.ExterQual.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    dataset.BsmtExposure = dataset.BsmtExposure.map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
    dataset.ExterCond = dataset.ExterCond.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    dataset.HeatingQC = dataset.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    dataset.Electrical = dataset.Electrical.map({'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5})
    dataset.FireplaceQu = dataset.FireplaceQu.map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    dataset.GarageQual = dataset.GarageQual.map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    dataset.GarageCond = dataset.GarageCond.map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
    dataset.BsmtFinType2 = dataset.BsmtFinType2.map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
    dataset.Functional = dataset.Functional.map({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8})
    dataset.BsmtFinType1 = dataset.BsmtFinType1.map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})

# 2. Categorical Features
for col in cat_cols:
    for df in [train, test]:
        df[col] = pd.factorize(df[col])[0]


#################################################
#       4. Reduce dataframes                    #
#################################################
print(f'################ {colored("4. Reduce memory usage of all dataframs", "green")} #############')
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

print(f'################ {colored("5. Saving Csv files", "green")} #############')
full_path = os.getcwd()
train.to_csv(full_path + '/output/train_processed.csv', index=False)
test.to_csv(full_path + '/output/test_processed.csv', index=False)
