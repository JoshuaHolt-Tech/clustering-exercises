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

from env import get_connection


def scale_zillow(train, val, test, cont_columns, scaler_model = 1):
    """
    This takes in the train, validate and test DataFrames, scales the cont_columns using the
    selected scaler and returns the DataFrames.
    *** Inputs ***
    train: DataFrame
    validate: DataFrame
    test: DataFrame
    scaler_model (1 = MinMaxScaler, 2 = StandardScaler, else = RobustScaler)
    - default = MinMaxScaler
    cont_columns: List of columns to scale in DataFrames
    *** Outputs ***
    train: DataFrame with cont_columns scaled.
    val: DataFrame with cont_columns scaled.
    test: DataFrame with cont_columns scaled.
    """
    #Create the scaler
    if scaler_model == 1:
        scaler = MinMaxScaler()
    elif scaler_model == 2:
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    #Make a copy
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    
    #Fit the scaler
    scaler = scaler.fit(train[cont_columns])
    
    #Build the new DataFrames
    train_scaled[cont_columns] = pd.DataFrame(scaler.transform(train[cont_columns]),
                                                  columns=train[cont_columns].columns.values).set_index([train.index.values])

    val_scaled[cont_columns] = pd.DataFrame(scaler.transform(val[cont_columns]),
                                                  columns=val[cont_columns].columns.values).set_index([val.index.values])

    test_scaled[cont_columns] = pd.DataFrame(scaler.transform(test[cont_columns]),
                                                 columns=test[cont_columns].columns.values).set_index([test.index.values])
    #Sending them back
    return train_scaled, val_scaled, test_scaled



def train_validate(df, stratify_col = None, random_seed=1969):
    """
    This function takes in a DataFrame and column name for the stratify argument (defualt is None).
    It will split the data into three parts for training, testing and validating.
    """
    #This is logic to set the stratify argument:
    stratify_arg = ''
    if stratify_col != None:
        stratify_arg = df[stratify_col]
    else:
        stratify_arg = None
    
    #This splits the DataFrame into 'train' and 'test':
    train, test = train_test_split(df, train_size=.7, stratify=stratify_arg, random_state = random_seed)
    
    #The length of the stratify column changed and needs to be adjusted:
    if stratify_col != None:
        stratify_arg = train[stratify_col]
        
    #This splits the larger 'train' DataFrame into a smaller 'train' and 'validate' DataFrames:
    train, validate = train_test_split(train, test_size=.4, stratify=stratify_arg, random_state = random_seed)
    return train, validate, test


def train_val_test(train, val, test, target_col):
    """
    Seperates out the target variable and creates a series with only the target variable to test accuracy.
    """
    #Seperating out the target variable
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_val = val.drop(columns = [target_col])
    y_val = val[target_col]

    X_test = test.drop(columns = [target_col])
    y_test = test[target_col]
    return X_train, y_train, X_val, y_val, X_test, y_test


def outlier_ejector(dataframe, column, k=1.5):
    """
    This function takes in a dataframe and looks for upper outliers.
    """
    q1, q3  = dataframe[column].quantile(q=[0.25, 0.75])
    iqr = q3 - q1
    
    
    lower_bound = q1 - (k * iqr)
    upper_bound = q3 + (k * iqr)
    
    high_items = dataframe[column] > upper_bound
    low_items = dataframe[column] < lower_bound

    
    return dataframe[~low_items & ~high_items]

def outlier_detector(dataframe, column, k=1.5):
    """
    This function takes in a dataframe and looks for upper outliers.
    """
    q1, q3  = dataframe[column].quantile(q=[0.25, 0.75])
    iqr = q3 - q1
    
    
    lower_bound = q1 - (k * iqr)
    upper_bound = q3 + (k * iqr)
    
    high_items = dataframe[column] > upper_bound
    low_items = dataframe[column] < lower_bound

    
    return dataframe[low_items & high_items]

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

def prep_data(df):
    """
    Target column should be in a yes/no, True/False, 0/1 format.
    This function is not designed to handle null values.
    Input DataFrame.
    Outputs a DataFrame with binary 
    columns as 0/1 and dummy columns.
    """
    
    #Variable
    dumb_columns = []

    #Values that will be turned to an integer of  0 or 1
    values_to_encode = {'Yes': 1, 'yes': 1, 'y': 0, 'Y': 1,
                      True : 1, 'T': 1, 'True': 1, 't': 1,'true': 1,
                      'No': 0, 'no': 0, 'n': 0, 'N' : 0,
                      False : 0, 'F': 0, 'f': 0, 'False': 0, 'false':0,
                       '0': 0, '1': 1}

    #Seperate out object and bool data type columns into new df:
    object_df = df.select_dtypes(include=['object','bool'])
    
    #For loop to find applicable columns
    for col in object_df:
        change = False

        #Filter to check if the values are the correct length and in the values_to_encode dict
        if (len(object_df[col].value_counts()) == 2):
            for item in object_df[col].unique():
                if item in values_to_encode.keys():
                    change = True

            #Swaps out old column with the new binary column
            if change == True:
                df = df.drop(columns=col)
                df = pd.concat([df, object_df[col].replace(to_replace=values_to_encode).astype('int')],
                               axis=1)                
            else:
                dumb_columns.append(object_df[col].name)
            change = False

        #Create dummy values for columns with < 6 unique values:        
        elif (len(object_df[col].value_counts()) < 6 ):
            dumb_columns.append(object_df[col].name)
            
    #Creates dummy columns based on list 'dumb_columns' and drops dummy source columns
    dummy_df = pd.get_dummies(object_df[dumb_columns])
    df = pd.concat([df, dummy_df], axis=1)
    df.drop(columns=dumb_columns, inplace = True)
    
    return df



def handle_missing_values(df, prop_required_column = .4, prop_required_row = .25):
    """
    This function drops columns then rows which contain a certain amount of null values.
    """
    #Lists to hold values
    drop_cols = []
    drop_rows = []
    na_cols_not_drop = ['taxdelinquencyyear']
    
    #Finds columns with lots of na values
    for col in df:
        if (df[col].isna().sum()/len(df) > prop_required_column):
            if col in na_cols_not_drop:
                pass
            else:
                drop_cols.append(f'{col}')
    #Drops columns with lots of na values        
    df = df.drop(columns=drop_cols)
    num_rows = int(len(df.columns) * prop_required_row)
    #Drops rows with lots of na values
    df = df.dropna(thresh=num_rows) 
    
    return df