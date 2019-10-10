from termcolor import colored
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

############### Show colored text #############
def bg(value, type='num', color='blue'):
    value = str('{:,}'.format(value)) if type == 'num' else str(value)
    return colored(' '+value+' ', color, attrs=['reverse', 'blink'])


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
            
############### Summary Table #####################
from scipy import stats

# Summary dataframe
def summary(df, sort_col=0):
    summary              = pd.DataFrame({'dtypes': df.dtypes}).reset_index()
    summary.columns      = ['Name', 'dtypes']
    summary['Missing']   = df.isnull().sum().values
    summary['M_Percent'] = round(100 * summary['Missing'] / df.shape[0], 2)
    summary['Uniques']   = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2), 2)

    summary = summary.sort_values(by=[sort_col], ascending=False) if sort_col else summary

    # Print some smmaries.
    print(f'~> Dataframe has {bg(df.shape[0])} Rows, and {bg(df.shape[1])} Columns.')
    print(f'~> Dataframe has {bg(summary[summary["Missing"] > 0].shape[0], color="red")} Columns have [Missing] Values.')
    print('---'*20)
    for type_name in np.unique(df.dtypes):
        print(f'~> There are {bg(df.select_dtypes(type_name).shape[1])}\t Columns that have [Type] = {bg(type_name, "s", "green")}')


    return summary.style.background_gradient(cmap='summer_r')
              
              
############## Plot Feature Imporances ########################
def plot_feature_importances(df, n = 10, threshold = None):
    """Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances. 

    Args:
        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".

        n (int): Number of most important features to plot. Default is 15.

        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.

    Returns:
        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 
                        and a cumulative importance column

    Note:

        * Normalization in this case means sums to 1. 
        * Cumulative importance is calculated by summing features from most to least important
        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance

    """
    plt.style.use('fivethirtyeight')

    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)

    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    plt.rcParams['font.size'] = 12

    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = sns.color_palette()[0], 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'{n} Most Important Features', size = 18)
    plt.gca().invert_yaxis()


    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);

        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))

        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();

        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))

    return df
              
              
              
              
              
              
########################## Show Annotation ###################
def show_annotation(dist, n=5, total=None):
    sizes = [] # Get highest value in y
    for p in dist.patches:
        height = p.get_height()
        sizes.append(height)

        dist.text(p.get_x()+p.get_width()/2.,          # At the center of each bar. (x-axis)
               height+n,                            # Set the (y-axis)
               '{:1.2f}%'.format(height*100/total) if total else '{}'.format(height), # Set the text to be written
               ha='center', fontsize=14) 
    dist.set_ylim(0, max(sizes) * 1.15); # set y limit based on highest heights