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
|  annual_salary_2016 |Int64    | the employee's current base salary would be for a year based on the position held at September 30, 2016   |
|   base_pay_2016   | Int64 |fixed amount of money that an employee receives prior to any extras being added or payments deducted |
|  leave_payout_2016  | Int64 |leave the employee sold back to the City during the fiscal year |
|  other_2016 | Int64 |reflects various incentives paid to City employees based on their job position and education level.|
|  overtime_2016  |  Int64  | compensatino earned through time worked past normal house  |
| gross_earnings_2016  | Int64 | total pay employees received between October 1st and September 30th.|
|  additional_compensation  |  Int64  |incremental costs incurred by the City of behalf of employees to include the employer's share of FICA/Medicare, TMRS [Pension], annual Health Assessment [average healthcare benefit costs] and other related fringe provided to employees.   |
|  total_compensation | Int64 |total cost the City incurred for the employee's services received between October 1st and September 30th|
| department   | object   | department that employee works in  |
|  gender | object |gender of employee|
|  ethnicity  |  object  | ethnicty of employee as reported to the EEOC  |
|  years_employed  |  Int64  | took todays date and substraced the hire date from it to get a 'years employed' feature  |
|  job_id  | object   | id of employees job_name  |
|  job_name  |  object  | job_name of employee  |
|   ethnicity_ASIAN |  uint8  | dummies columns where 1 = Asian & 0 = not  |
|  ethnicity_BLACK  |  uint8  |  dummies columns where 1 = Black & 0 = not   |
|  ethnicity_HISPANIC  |  uint8  | dummies columns where 1 = Hispanic & 0 = not    |
|  ethnicity_NATIVE AMERICAN/ALASKAN  |  uint8  | dummies columns where 1 = Native American/Alaskan & 0 = not    |
|  ethnicity_NATIVE HAWAIIAN  |  uint8  | dummies columns where 1 = Native Hawaiian & 0 = not    |
|  ethnicity_OTHER  | uint8   | dummies columns where 1 = Other & 0 = not    |
|  ethnicity_WHITE  |  uint8  | dummies columns where 1 = White & 0 = not    |





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
 explore.py
 model.py
 cosa_employee_salary.csv or you can acquire it from data.world ---> https://data.world/amillerbernd/san-antonio-city-salary-data
 
 Your target variable will be gender which is defined in the above data dictionary. Please visit my final notebook to see this variable being used.
 
 <b>Step 0</b> Clone this repo to your local machine with the above files.
 
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