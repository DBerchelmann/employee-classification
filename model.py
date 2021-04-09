import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from graphviz import Graph
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from wrangle import new_city_data, clean_city, missing_zero_values_table, train_validate_test_split






def run_model():
   
    df = new_city_data()
    
    #clean the data
    
    df = clean_city(df)

    # train, validate, split the data
    
    train, validate, test = train_validate_test_split(df, seed=123)
    
    
    
    # Select features to be used in the model
    cols = ['annual_salary_2016','base_pay_2016', 'leave_payout_2016', 'other_2016', 'overtime_2016', 'additional_compensation', 'total_compensation',
           'years_employed', 'ethnicity_ASIAN', 'ethnicity_BLACK', 'ethnicity_HISPANIC', 'ethnicity_NATIVE AMERICAN', 'ethnicity_NATIVE HAWAIIAN',
           'ethnicity_OTHER', 'ethnicity_WHITE']

    X = test[cols]
    y = test.gender
    
    # Create and fit the model
    forest = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=1,
                            n_estimators=100,
                            max_depth=10, 
                            random_state=123).fit(X, y)

    # Create a DataFrame to hold predictions
    results = pd.DataFrame(
        {'Actual_Gender': test.gender,
         'Model_Predictions': forest.predict(X),
         'Model_Probabilities': forest.predict_proba(X)[:,1]
        })

    # Generate csv
    results.to_csv('model_results.csv')

    return results