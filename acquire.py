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




def wrangle_zillow():
    """
    This function reads the zillow data from Codeup db into a df.
    Changes the names to be more readable.
    Drops null values.
    """
    filename = "full_zillow_2017.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, parse_dates=['transactiondate'])
    else:
        
        # read the SQL query into a dataframe
        query = """
        SELECT * FROM properties_2017
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN predictions_2017 USING (parcelid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        LEFT JOIN unique_properties USING (parcelid)
        WHERE transactiondate LIKE "2017%%";
        """

        df = pd.read_sql(query, get_connection('zillow'))
        df['latitude'] = df['latitude'] / 10_000_000
        df['longitude'] = df['longitude'] / 10_000_000
        df.drop(columns-'id', inplace=True)
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        
        # Return the dataframe to the calling code
        return df
    
def wrangle_mall():
    """
    This function gets all data from the mall_customers database.
    """
    filename = "mall_customers.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        
        # read the SQL query into a dataframe
        query = """
        SELECT * FROM customers;
        """

        df = pd.read_sql(query, get_connection('mall_customers'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        
        # Return the dataframe to the calling code
        return df
    


def wrangle_iris():
    """
    This function gets all data from the iris database.
    """
    filename = "iris_db.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        
        # read the SQL query into a dataframe
        query = """
        SELECT * FROM measurements 
        LEFT JOIN species USING (species_id);
        """

        df = pd.read_sql(query, get_connection('iris_db'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        
        # Return the dataframe to the calling code
        return df