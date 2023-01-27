import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error


#Removes warnings and imporves asthenics
import warnings
warnings.filterwarnings("ignore")


def explore_relationships(feature_list, train, target_col, visuals = False):
    """
    This function takes in a list of features, grabs the .describe() metrics associated with the target column.
    *** Inputs ***
    feature_list: List of DataFrame column names to iterate through and compare to target column.
    train: Panda's DataFrame to explore.
    target_col: String. Title of target column.
    *** Output ***
    DataFrame with metrics to explore
    """
    metrics = []
    for feature in feature_list:
        num_items = train[feature].unique()
        num_items.sort()
        for item in num_items:
            temp_df = train[train[feature] == item][target_col].describe()
            temp_metrics = {
                'comparison' : f'{item}_{feature}',
                'count' : round(temp_df[0],0),
                'mean' : round(temp_df[1],0),
                'std' : round(temp_df[2],0),
                'min' : round(temp_df[3],0),
                '25%' : round(temp_df[4],0),
                '50%' : round(temp_df[5],0),
                '75%' : round(temp_df[6],0),
                'max' : round(temp_df[7],0)}
            metrics.append(temp_metrics)

    feature_per_item = pd.DataFrame(metrics)
    if visuals == True:
        sns.lineplot(data=feature_per_item, x='comparison', y='25%',
                             legend='brief').set(title=f'{target_col} to {feature} comparison',
                                                 xlabel =f'{feature}', ylabel = f'{target_col}')
        sns.lineplot(data=feature_per_item, x='comparison', y='mean', markers=True)
        sns.lineplot(data=feature_per_item, x='comparison', y='50%')
        sns.lineplot(data=feature_per_item, x='comparison', y='75%')
        plt.ylabel(f'{target_col}')
        plt.xlabel(f'{item}_{feature}')
        
    return feature_per_item

def find_na(df):
    list_of_na = []
    for col in df:
        temp_dict = {'column_name': f'{col}' , 
                     'num_rows_missing': df[col].isna().sum(),
                     'unique_values': df_sorted[col].value_counts().sum(),
                     'pct_rows_missing': round(df[col].isna().sum() / len(df[col]),5)
                     }

        list_of_na.append(temp_dict)

    na_df = pd.DataFrame(list_of_na)
    na_df.set_index('column_name')
    return na_df

def correlation_test(df, target_col, alpha=0.05):
    """
    Maybe create a function that automatically seperates continuous from discrete columns.
    """
    
    list_of_cols = df.select_dtypes(include=[int, float]).columns
              
    metrics = []
    for col in list_of_cols:
        
        #Checks skew to pick a test
        if abs(df[target_col].skew()) > 0.5 or abs(df[col].skew()) > 0.5:
            corr, p_value = stats.kendalltau(df[target_col], df[col], nan_policy='omit')
            test_type = 'Spearman R'
        else:
            # I'm unsure how this handles columns with null values in it.
            corr, p_value = stats.pearsonr(df[target_col], df[col])
            test_type = 'Pearson R'

        #Answer logic
        if p_value < alpha:
            test_result = 'relationship'
        else:
            test_result = 'independent'

        temp_metrics = {"Column":col,
                        "Correlation": corr,
                        "P Value": p_value,
                        "Test Result": test_result}
        metrics.append(temp_metrics)
    distro_df = pd.DataFrame(metrics)              
    distro_df = distro_df.set_index('Column')

    #Plotting the relationship with the target variable (and stats test result)
    my_range=range(1,len(distro_df.index) + 1)
    hue_colors = {'relationship': 'green', 'independent':'red'}

    plt.figure(figsize=(6,9))
    plt.axvline(0, c='tomato', alpha=.6)

    plt.hlines(y=my_range, xmin=-1, xmax=1, color='grey', alpha=0.4)
    sns.scatterplot(data=distro_df, x="Correlation",
                    y=my_range, hue="Test Result", palette=hue_colors,
                    style="Test Result")
    plt.legend(title="Stats test result")

    # Add title and axis names
    plt.yticks(my_range, distro_df.index)
    plt.title(f'Statistics tests of {target_col}', loc='center')
    plt.xlabel('Neg Correlation            No Correlation            Pos Correlation')
    plt.ylabel('Feature')
    
    #Saves plot when it has a name and uncommented
    #plt.savefig(f'{train.name}.png')