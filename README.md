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

Using the processed data, the model is built in [AlphabetSoupCharity.ipynb](https://github.com/caseylee94/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb) using `tensorflow.keras.models.Sequential` and `tensorflow.keras.layers.Dense` with the following parameters:

| Parameter | Value | Explanation |
| --------- | ----- | ------------- |
| Hidden Layers | 2 | Key component that enables neural network to learn complex tasks; 2 layers are a good starting point considering the data has low complexity and offers a relatively short computation time |
| Number of Nodes in Layers | 80, 30 | First layer has two times the input number (43), second layer has smaller number for shorter computation time | 
| Hidden Layer Activation Function | ReLu | Standard function that generally produces good results with a short computation time; it is simple to implement and effective, is a good starting point for this model |
| Number of Output Layers | 1 | Binary classification model only needs one output layer |
| Output Layer Activation Function | Sigmoid | Function used to predict probability in binary classification outputs |

*Figure 1: Table showing parameters for initial neural network model*

Using this model, the accuracy score was found to be 73.47%. This is below our target accuracy score of 75%, so optimization of the model was attempted next to try to raise this score by changing one feature at a time while holding the other features fixed. These models are built in [AlphabetSoupCharity_Optimization.ipynb](https://github.com/caseylee94/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb)

| Optimization Method | Accuracy |
| ------------------- | -------- |
| Changed Number of Epochs from 100 to 200 | 73.70% |
| Sigmoid Activation Function in the Hidden Layer Instead of ReLU | 73.33% |
|Increasing the Number of Hidden Layers | 73.59% |
| Adjusting Input Data by Dropping or Binning Noisy Variables | 72.66% |

*Figure 2: Table showing accuracy of each optimization attempt*

### Summary

Using the inital model and optimization techniques, the model was not able to meet the target accuracy of 75%. Moving forward, changing the model type is a good next step to try to achieve this accuracy. A `Random Forest` machine learning model works well for classification problems and is comparable to a deep neural network model with two hidden layers. It is a supervised machine learning model that is relatively easy to build and utilize; it does not require as many features and inputs as the neural network model so it could be tried without much time commitment.

Alternatively, given more time, more optimization methods can be attempted on this model. Raising the number of epochs resulted in a higher accuracy so this could be attempted again with a higher amount, potentially 300 epochs. Increasing the number of hidden layers also resulted in a higher accuracy score; this optimization could be attempted in congruence with raising the number of epochs. Again, this model will take more to run and could be attempted given more future resources.
