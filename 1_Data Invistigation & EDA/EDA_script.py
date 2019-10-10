'''
Author:      Mosaab Muhammed.
Date:        10/10/2019
Title:       EDA & Data Investigation & NaN Filling.
Description: The aim of this script is to fill in the 'NaN' values, and
			 output that results to ./output/ folder.
'''
##################################
#      1.1 Import Libraries:      #
##################################
from termcolor import colored
print(f'################ {colored("1. Importing Libraries", "green")} ################')
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

#######################################
#    1.2 Some Utility Functions:      #
#######################################
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


##################################
#       2. Read Data             #
##################################
print(f'################ {colored("2. Reading Data", "green")} #############')
train = pd.read_csv('../0_Data/train.csv')
test = pd.read_csv('../0_Data/test.csv')

for data in [train, test]:
    print(f'~> [{var2str(data).ljust(5)}] has {bg(data.shape[0])} rows, and {bg(data.shape[1])} columns.')

##################################
#      3. Filling NaN values     #
##################################
print(f'################ {colored("3. Drop semi-constant columns", "green")} #############')
to_drop = ['Utilities', 'Street', 'PoolArea', 'MiscVal']
for df in [train, test]:
    df.drop(to_drop, axis=1, inplace=True)

##################################
#      4. Filling NaN values     #
##################################
print(f'################ {colored("4.1 Filling NaN values for Categorical Features", "green")} #############')
for df in [train, test]:
    df.Alley.fillna('None', inplace=True)
    df.MasVnrType.fillna('None', inplace=True)
    df.BsmtQual.fillna('None', inplace=True)
    df.BsmtCond.fillna('None', inplace=True)
    df.BsmtExposure.fillna('None', inplace=True)
    df.BsmtFinType1.fillna('None', inplace=True)
    df.BsmtFinType2.fillna('None', inplace=True)
    df.FireplaceQu.fillna('None', inplace=True)
    df.GarageType.fillna('None', inplace=True)
    df.GarageFinish.fillna('None', inplace=True)
    df.GarageQual.fillna('None', inplace=True)
    df.GarageCond.fillna('None', inplace=True)
    df.GarageCars.fillna(0, inplace=True)
    df.PoolQC.fillna('None', inplace=True)
    df.Fence.fillna('None', inplace=True)
    df.MiscFeature.fillna('None', inplace=True)
    df.MSZoning.fillna('RL', inplace=True)
    df.Exterior1st.fillna(df.Exterior1st.mode()[0], inplace=True)  # Fill it with the most common one
    df.Exterior2nd.fillna(df.Exterior2nd.mode()[0], inplace=True)
    df.BsmtFinSF1.fillna(0, inplace=True)
    df.BsmtFinSF2.fillna(0, inplace=True)
    df.BsmtUnfSF.fillna(0, inplace=True)
    df.TotalBsmtSF.fillna(0, inplace=True)
    df.BsmtFullBath.fillna(0, inplace=True)
    df.BsmtHalfBath.fillna(0, inplace=True)
    df.KitchenQual.fillna(df.KitchenQual.mode()[0], inplace=True)
    df.Functional.fillna('Typ', inplace=True)
    df.SaleType.fillna(df.SaleType.mode()[0], inplace=True)


print(f'################ {colored("4.2 Filling NaN values for Numerical Features", "green")} #############')
# Fill NaN for Numerical Features.
for df in [train, test]:
    df.LotFrontage.fillna(df.LotFrontage.median(), inplace=True)  # LotFrontage
    df.Electrical.fillna('SBrkr', inplace=True) 				  # Electrical
    df.MasVnrArea.fillna(0, inplace=True)  					      # MasVnrArea
    df.GarageYrBlt.fillna(0, inplace=True)  					  # GarageYrBlt
    df.GarageArea.fillna(0, inplace=True)  					      # GarageArea

if (train.isnull().sum().sum() == 0) and (test.isnull().sum().sum() == 0):
    print('~> There is no columns have missing values!\n~> Saving the resultant csv datasets to ./output.')
    full_path = os.getcwd()
    train.to_csv(full_path + '/output/train_null_removed.csv', index=False)
    test.to_csv(full_path + '/output/test_null_removed.csv', index=False)
else:
    train_null_cols = [col for col in train.columns if train[col].isnull().sum() > 0]
    test_null_cols = [col for col in test.columns if test[col].isnull().sum() > 0]
    print(f"~> There are {bg(len(train_null_cols+test_null_cols), color='red')} columns have missing values!\n~> Saving files aborted!")
    print(train_null_cols)
    print('--' * 20)
    print(test_null_cols)
