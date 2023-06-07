# Group H: Python Assessment


## Description
In this data science project, our team of five students will work collaboratively to load and analyze and perform machine learning a selected health care dataset. 
We will leverage the power of Python libraries such as pandas, numpy, seaborn, geopandas, sklearn, scipy and matplotlib to accomplish our objectives.

Data - https://www.kaggle.com/datasets/thedevastator/home-health-care-agency-ratings
Supplementary Data - https://drive.google.com/drive/folders/1DUU2hL4PAax-QpcpMiHJ-EOb6sRe-PbC?usp=sharing


### Members
    •	Etoroabasi Akpan - 		    S4211614	
    •	Enitan Sulaimon Ogungbemi -	S4219042
    •	Kingsley Ezeogwum - 		S4212037
    •	Anyanacho Obed Johua -  	S4212171
    •	Raaquib Qureshi - 	        S4217002
    
   

## Installation
To run this project, please ensure that Python is installed on your machine. You can install Python by downloading it from the official Python website (https://www.python.org/) or by using the Anaconda distribution, which provides a comprehensive Python environment with many useful packages pre-installed.

Please make sure to install a compatible version of Python (version 3.9 and above). 

Once Python is installed, you can proceed with setting up the project environment and installing any necessary dependencies as mentioned in the project instructions.

### Dependencies
The following dependencies and libraries are required to run the project:

NumPy version: 1.24.3  -used for numerical manipulations
Pandas version: 1.5.3 - used for data manipulations and reading files
SK Learn: 1.2.2       - used as a  library providing several machine learning functionalities
matplotlib version: 3.3.4 - used for visualization
scipy version 1.10.1  - used for additional data and numerical manipulations
missing no version 0.5.2 - used to visualize missing numbers
geopandas version 0.13.0 - used to imjest and manipulate geospatial data
seaborn version 0.11.1 - used for visualization
xgboost version 1.7.5 - used for XGboost analysis
statsmodel version 0.13.5 - used for statisitical modelling and testing 

You can install these dependencies by running the following command:
```pip install matplotlib numpy pandas plotly scipy seaborn scikit-learn```


## Data

For this project, we opted to utilize data of care homes in United States of America. The is updated as at 2016 and contains 11366 rows and 64 columns

By utilizing this dataset, we can gain valuable insights into the operations of each Home Health Care Agency. It allows us to make informed decisions regarding care needs and assess the quality measure ratings of different agencies.

This information empowers individuals to make well-informed choices about their healthcare options. Whether it's dedicated nursing care services, speech pathology, or medical social services. It is diviided into 5 broad aspects:

1. Informational Details of the Care Home
2. Type of Service Offered by Care Home
3. Rating of the Care Home Health team based on several performance matrices
4. Rating of the Care Home Patient's improvement based on several matrices
5. Comparison of Care Home key figures with National Average

Download Link: https://www.kaggle.com/datasets/thedevastator/home-health-care-agency-ratings

The data downloaded is a csv file and shoild be saved in the same directory as your code file


## Suppleentary Data

We harnessed some supplementary data to gather additional features for our data set. They include:

col_rename.csv - A csv file to rename our data columns.

data/usa-states-census-2014.shp - shape file stored in a data folder to visualize geographical region in USA

Both files can be downloaded from the link: https://drive.google.com/drive/folders/1DUU2hL4PAax-QpcpMiHJ-EOb6sRe-PbC?usp=sharing

The csv file and data folder should be stored in the same directory as your code file






## Project Steps

The project is divided into 8 sections

## Introduction

Here we give an introduction to the project and what it would entail

## Importing libraries
Here we imported all the relevant librries (listed earlier) required for our code to run smoothly

```python
    import numpy as np
    import pandas as pd
    import missingno as msno
```

## Reading Data

Here we read in our data set and performed basic descriptive exploration around it

```python
        raw_data  = pd.read_csv('csv-1.csv')
        raw_data.head()
```

## Data Cleaning

Here we performed several data cleaning tasks such filling missing values, column removal, outlier detection etc

```python
        def missing_data (raw_data):
    
    missing_data_df = pd.DataFrame(columns = ['column_name', 'data_type', 'no_of_missing_data', 'percent_missing'])
    
    for column in raw_data.columns:
        
        data_type = str(raw_data[column].dtype)
        
        null_values = raw_data[column].isnull().sum()
    
        null_percent = raw_data_copy[column].isnull().mean()*100
        
        null_percent = round(null_percent, 2)

        row = pd.Series([column, data_type, null_values, null_percent], index = missing_data_df.columns)
        
        row = row.to_frame().T
        
        #row = pd.Series([column, data_type, null_values, null_percent])
    
        missing_data_df = pd.concat([missing_data_df, row], ignore_index = True)
        
    
    #return row
    missing_data_df = missing_data_df.sort_values(by = ['no_of_missing_data'], ascending = False)
        
    return missing_data_df
```


## Exloratory Data Analysis

Here, we explored our cleaned data set to extablish relationships between the variables.
Some of the tasks performed here are univariate analysis, bivariate analysis, correlation and hypothesis test


```python
    data_grouped = raw_data_copy.groupby('ownership_type')['rat_pt_care_qual'].mean()
    data_grouped.plot(kind='bar', color='blue')
    plt.title('Average Patient Care Quality Rating by Ownership Type')
    plt.xlabel('Ownership Type')
    plt.ylabel('Average Patient Care Quality Rating')
    plt.show()
```


### Feature Selection

At this stage, we performed several tests and evaluations to select the most significant features from our data set for predictive analysis. Some of the tasks performed
here are train-test split, cramer's V analysis, VIF

```python
    def vif_score(data):

        numerical_features = data.select_dtypes(include = ['number', 'bool'])  #select only numeric data

        vif_data = pd.DataFrame()
        vif_data["Feature"] = numerical_features.columns
        vif_data["VIF"] = [variance_inflation_factor(numerical_features.values, i) for i in range(numerical_features.shape[1])]

        return vif_data

```



## Machine Learning

Here, we implemented the various ML techniques to model our data to predict outcomes

The models performed in this projects are 
- Linear Regression
- Random Forest
- XGBoost

```python
        dtrain = xgb.DMatrix(data=features_train_xg, label=target_train, enable_categorical=True)

        dtest  = xgb.DMatrix(data=features_test_xg, label=target_test, enable_categorical=True)


        params = {
            'objective': 'reg:squarederror',
            'boosting_type': 'gbtree',
            'learning_rate': 0.16,
            'n_estimators': 700,
            #'max_depth': 5000,
            'early_stopping_rounds':100
        }


model = xgb.train(params, dtrain)
```

## Evaluation

Here, we evaluated the various machine learning models which we implemented in the previous section and  made recommendations based  on the evalauation


```python

def evaluate_model(target_test, target_prediction):
    

    mse = mean_squared_error(target_test, target_prediction)
    mae = mean_absolute_error(target_test, target_prediction)
    r2 = r2_score(target_test, target_prediction)


    n = features_train.shape[0]
    # Number of features (predictors, p) is the shape along axis 1
    p = features_train.shape[1]

    # We find the Adjusted R-squared using the formula
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
    return [mse, mae, r2, adjusted_r2]

```


## Conclusion

We concluded that out of the over 60 initial features, only 14 were significant in predicting the overall rating of a care home
From the extensive analysis performed in this project, we could conclude that Random Forest was the best Machine Learning model to perform predictions on the care home data set.
