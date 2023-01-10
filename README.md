# Neural_Network_Charity_Analysis

## Overview of Project

This project is aimed to help a pseudocompany, Alphabet Soup, that donates money to businesses for charity work, determine which businesses are worth donating to and which are too high risk. Sometimes these companies take the money and do impactful charity work as promised, and sometimes they take the money and disappear. To predict which choice a new company is likely to make, deep neural network modeling will be utilized to build a binary classification model. The model takes in nine features from a loan application data set about past companies that have accepted money from Alphabet Soup. A `Tensorflow Keras Sequential model` with `dense hidden layers` will be employed and optimized with a goal of achieving greater than 75% accuracy of predictions.

## Resources
* Dataset: [charity_data.csv](https://github.com/caseylee94/Neural_Network_Charity_Analysis/tree/main/Resources)
* Software:
    * Python 3.7.6
    * scikit-learn 0.22.1
    * pandas 1.0.1
    * TensorFlow 2.4.1
    * NumPy 1.19.5
    * Matplotlib 3.1.3
    * Jupyter Notebook 1.0.0

## Analysis

### Data Preprocessing

The first step in this project was to read in the csv file as a dataframe, using `Pandas`. Any unnecessary columns were then dropped, in this case the "EIN" and "NAME" column are dropped because these are for identification of the different companies and do not contribute impactful information for a neural network model. Then, the target and feature variables are determined as follows:

* Target variable: `IS_SUCCESSFUL`
* Feature variables: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`

The variables are checked for unique values and any noisy variables are bucketed, in this first analysis this was `CLASSIFICATION` and `APPLICATION_TYPE`, each with 15+ unique values bucketed down to 8 or less values. Then, any categorical data within the variables are encoded using `sklearn.preprocessing.OneHotEncoder`.  The data are split into training and testing sets and then scaled using `sklearn.preprocessing.StandardScaler` to optimize the model; this ensures data are within a certain range that helps the model compare the data, as well as run faster and more efficiently.

### Building, Training, and Optimizing the Model


