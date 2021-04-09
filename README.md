<h1> Individual Project - Employee Classification </h1>

Hi there,

Welcome to the README file for my individual project covering <b>classification of gender based on 2016 San Antonio City Employee Compensation</b>

In here, you will find expanded information on this project including goals, how I will be working through the pipeline and a data dictionary to help offer more insight to the variables that are being used.

-------------------
<h3><u>The Goal</u></h3>

<font color = blue>**Why are we here?**</font>

* <font color = red>Goal 1:</font> <i>Create a model to predict whether an employee is a male or female based on 2016 salary data from the city of San Antonio</i>
* <font color = red>Goal 2:</font> <i>Compare compensation based on gender and race</i>

-------------------
<h3><u>Where Is Our Data Coming From?</u></h3>

* This data is being pulled from data.world.
    * It can be found at the following link. https://data.world/amillerbernd/san-antonio-city-salary-data

* This repository also has a CSV of the data available as well

------------------
<H3><u> Project Planning </u></H3>

In addition to this README, you can see my TRELLO project pipeline by visiting the following link: https://trello.com/b/RCVreIQS/individual-project-salary-classification

Here is a snapshot of my project planning/setup on the afternoon of 4/8/21
 
<img src="https://i.ibb.co/kcwGGYP/trello-board.png" alt="trello-board" border="0">

-------------

<h3><u>Data Dictionary</u></h3>
    
-  Please use this data dictionary as a reference for the variables used within in the data set.



|   Feature      |  Data Type   | Description    |
| :------------- | :----------: | -----------: |
|  annual_salary_2016 |Int64    |    |
|   base_pay_2016   | Int64 |fixed amount of money that an employee receives prior to any extras being added or payments deducted |
|  leave_payout_2016  | Int64 | |
|  other_2016 | Int64 ||
|  overtime_2016  |  Int64  |   |
| gross_earnings_2016  | Int64 ||
|  additional_compensation  |  Int64  |   |
|  total_compensation | Int64 ||
| department   | object   |   |
|  gender | object ||
|  ethnicity  |  object  |   |
|  years_employed  |  Int64  |   |
|  job_id  | object   |   |
|  job_name  |  object  |   |
|   ethnicity_ASIAN |  uint8  |   |
|  ethnicity_BLACK  |  uint8  |   |
|  ethnicity_HISPANIC  |  uint8  |   |
|  ethnicity_NATIVE AMERICAN/ALASKAN  |  uint8  |   |
|  ethnicity_NATIVE HAWAIIAN  |  uint8  |   |
|  ethnicity_OTHER  | uint8   |   |
|  ethnicity_WHITE  |  uint8  |   |





-------------------
 <h3><u>Hypothesis and Questions</u></h3>

- Is salary correlated with gender?
- Is salary correlated with ethinicty?
- Is years employed correlated with salary?
- Is years employed correlated with gender?
- Is year employed correlated with ethnicity?
- Is additional compensation correlated with ethnicity?
- Is additional compensation correlated with gender?

<h5> The questions above will be answered using correlation tests, chi^2 tests and t-tests.</h5>

--------------------
 <h3><u>How To Recreate This Project</u></h3>
 
 To recreate this project you will need use the following files:
 
 wrangle.py
 evaluate.py
 explore.py
 
 Your target variable will be tax_assessed_value which is defined in the above data dictionary. Please visit my final notebook to see this variable being used.
 
 <b>Step 1.</b> Import all necessary libraries to run functions. These can be found in each corresponding .py file
 
 <b>Step 2.</b> Use pd.read to bring in the csv file from your local folder. 
 
 <b>Step 3.</b> To see the the cleaned data set before training do the following:
 
```df = clean_df()``` 

After you have gotten to know the data set, run the following to gather the train, validate, test data


    
 
 <b>Step 4.</b> Verify that your data has been prepped using df.head()
 
 <b>Step 5.</b>. Enter the explore phase using the different univariate, bivariate, and multivariate functions from the explore.py file. This is also a great time to use different line plots, swarm plots, and bar charts. The associated libraries to make the charts happen are from matplotlib, seaborn, scipy, plotly and sklearn
 
 <b>Step 6.</b> Evaluate and model the data using different regression algorithms. 
         
         
 ```
 { 
 from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from graphviz import Graph
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
 }
 ```
 
<b>Step 7.</b> After you have found a model that works, test that model against out of sample data using the function in my notebook.
 
 For a more detailed look, please visit my final notebook for employee classification for further assistance.
 
--------------------