import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data
from datetime import date 
from scipy import stats

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

import sklearn

from sklearn.model_selection import train_test_split

#Create connection~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Create helper function to get the necessary connection url.
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
    
# Grab the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
    
    
def new_city_data():
    '''
    This function reads data from the Codeup db into a df.
    '''
    df = pd.read_csv('sa_employees.csv', index_col=0)
    
    return df




# Clean the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def clean_city(df):
    '''This functions cleans our dataset using a variety of tools:
    
    '''
    df = new_city_data()
    
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)
    
    # rename columns
    
    df.columns = ['first_name', 'middle_name', 'last_name', 'hire_date', 'annual_salary_2016', 'base_pay_2016', 'leave_payout_2016', 'other_2016', 'overtime_2016', 'gross_earnings_2016', 'additional_compensation', 'total_compensation', 'job_title', 'department', 'gender', 'ethnicity', 'dept_subgroup']
        
    # remove salary outliers
    df = df[df.annual_salary_2016 < 200_000]
    
    # Convert hire_date data type to datetime data type
    df['hire_date'] = pd.to_datetime(df['hire_date'])  
    
    # Create new column called year
    df['year'] = df['hire_date'].dt.year
    
    # Create a 'today' to calculate a new column called years_employed
    today = pd.to_datetime('today')
    df['years_employed'] = today.year - df['year']
    
    # Round columns down to just two decimal places
    cols = ['annual_salary_2016', 'base_pay_2016', 'leave_payout_2016', 'other_2016', 'overtime_2016', 'gross_earnings_2016', 'additional_compensation', 'total_compensation']
    df[cols] = df[cols].round(0)
    
        
    # seperate job_title into job_id & job name column using split
    
    df[['job_id','job_name']] = df.job_title.str.split(pat= "-", n=1, expand=True)
    
    # drop columns
    
    dropcols = ['first_name', 'middle_name', 'last_name', 'dept_subgroup', 'year', 'job_title', 'hire_date']
    
   
    # drop cols from above
    df.drop(columns=dropcols, inplace=True)
    
    # clean up ethnicity column and replace values with similiar groupings under one descriptor
    df['ethnicity'] = df['ethnicity'].replace({'HISPANIC OR LATINO': 'HISPANIC', 'WHITE (NON HISPANIC OR LATINO)': 'WHITE', 'BLACK OR AFRICAN AMERICAN (NON HISPANIC OR LATINO)': 'BLACK', 'ASIAN (NON HISPANIC OR LATINO)': 'ASIAN', 'ASIAN OR PACIFIC ISLANDER': 'ASIAN', 'AMERICAN INDIAN OR ALASKA NATIVE (NONHISPANIC/LAT)': 'NATIVE AMERICAN', 'TWO OR MORE RACES (NON HISPANIC OR LATINO)': 'OTHER', 'NATIVE HAWAIIAN/OTHER PACIFIC ISLANDER (NON HIS)': 'NATIVE HAWAIIAN', 'NATIVE AMERICAN/ALASKAN': 'NATIVE AMERICAN'})
    
    # create dummy columns of encoded categorical variables
    dummies = pd.get_dummies(df[['ethnicity']], drop_first=False)
    
    # combine the original data frame with the new dummies columns
    df = pd.concat([df, dummies], axis=1)
    
    cols = ['annual_salary_2016', 'base_pay_2016', 'leave_payout_2016', 'other_2016', 'overtime_2016', 'gross_earnings_2016', 'additional_compensation', 'total_compensation']
    
    # df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').convert_dtypes() 

    
    return df
    

# Train/Split the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split(df, stratify_by= None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(train, test_size=.3, random_state=123)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train[stratify_by])
    return train, validate, test


# Create X_train, y_train, etc...~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def seperate_y(train, validate, test):
    '''
    This function will take the train, validate, and test dataframes and seperate the target variable into its
    own panda series
    '''
    
    X_train = train.drop(columns=[''])
    y_train = train.logerror
    X_validate = validate.drop(columns=[''])
    y_validate = validate.logerror
    X_test = test.drop(columns=[''])
    y_test = test.logerror
    return X_train, y_train, X_validate, y_validate, X_test, y_test

# Scale the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def scale_data(X_train, X_validate, X_test):
    '''
    This function will scale numeric data using Min Max transform after 
    it has already been split into train, validate, and test.
    '''
    
    
    obj_col = []
    num_train = X_train.drop(columns = obj_col)
    num_validate = X_validate.drop(columns = obj_col)
    num_test = X_test.drop(columns = obj_col)
    
    
    # Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()
    
   
    # we only .fit on the training data
    scaler.fit(num_train)
    train_scaled = scaler.transform(num_train)
    validate_scaled = scaler.transform(num_validate)
    test_scaled = scaler.transform(num_test)
    
    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=num_train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=num_train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=num_train.columns)
    
    
    return train_scaled, validate_scaled, test_scaled

# Combo Train & Scale Function~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split_seperate_scale(df, stratify_by= None):
    '''
    This function will take in a dataframe
    seperate the dataframe into train, validate, and test dataframes
    seperate the target variable from train, validate and test
    then it will scale the numeric variables in train, validate, and test
    finally it will return all dataframes individually
    '''
    
    # split data into train, validate, test
    train, validate, test = split(df, stratify_by= None)
    
     # seperate target variable
    X_train, y_train, X_validate, y_validate, X_test, y_test = seperate_y(train, validate, test)
    
    
    # scale numeric variable
    train_scaled, validate_scaled, test_scaled = scale_data(X_train, X_validate, X_test)
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test, train_scaled, validate_scaled, test_scaled

# Classification Train & Scale Function~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_validate_test_split(df, seed=123):
    
    df = clean_city(df)
    
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.gender
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.gender,
    )
    return train, validate, test

# Miscellaneous Prep Functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''''''''''''''''''''
'                  '
' Helper Functions '
'                  '
''''''''''''''''''''



def missing_zero_values_table(df):
    
    '''This function will look at any data set and report back on zeros and nulls for every column while also giving percentages of total values
        and also the data types. The message prints out the shape of the data frame and also tells you how many columns have nulls '''
    
    
    
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    null_count = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, null_count, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'null_count', 2 : '% of Total Values'})
    mz_table['Total Zeroes + Null Values'] = mz_table['Zero Values'] + mz_table['null_count']
    mz_table['% Total Zero + Null Values'] = 100 * mz_table['Total Zeroes + Null Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] >= 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " +  str((mz_table['null_count'] != 0).sum()) +
          " columns that have NULL values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)

    return mz_table




def handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5):
    ''' function which takes in a dataframe, required notnull proportions of non-null rows and columns.
    drop the columns and rows columns based on threshold:'''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    
    
    return df


def features_missing(df):
    
    '''This function creates a new dataframe that analyzes the total features(columns) missing for the rows in
    the data frame. It also give s a percentage'''
    
    # Locate rows with. missing features and convert into a series
    df2 = df.isnull().sum(axis =1).value_counts().sort_index(ascending=False)
    
    # convert into a dataframe
    df2 = pd.DataFrame(df2)
    
    # reset the index
    df2.reset_index(level=0, inplace=True)
    
    # rename the columns for readability
    df2.columns= ['total_features_missing', 'total_rows_affected'] 
    
    # create a column showing the percentage of features missing from a row
    df2['pct_features_missing']= round((df2.total_features_missing /df.shape[1]) * 100, 2)
    
    # reorder the columns for readability/scanning
    df2 = df2[['total_features_missing', 'pct_features_missing', 'total_rows_affected']]
    
    return df2

def outlier_function(df, cols, k):
	#function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df.annual_income.quantile(0.25)
        q3 = df.annual_income.quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df