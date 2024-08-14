# Titanic Survival Prediction

This repository contains an analysis of the Titanic dataset, where the goal is to predict whether a passenger survived the Titanic disaster based on various features such as age, sex, and ticket class.



## Project Overview

The Titanic dataset is a classic machine learning dataset used to practice binary classification. In this project, we perform data exploration, preprocessing, and model training to predict the survival of passengers. The analysis is conducted using Python and various machine learning libraries.

### Key Steps
1. **Read Csv File:** read the csv file and do data exploration

2. **Data Exploration:** Understand the data by exploring its structure, missing values, and distributions.

3. **Data Preprocessing:** Handle missing values, encode categorical features, and split the data into training and testing sets.

4. **Model Training:** Train a machine learning model (e.g., Random Forest, Logistic Regression) to predict passenger survival.

5. **Model Evaluation:** Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.

## Dataset

The dataset used in this analysis is the Titanic dataset provided by [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data). It contains the following features:

- **PassengerId:** Unique ID for each passenger.
- **Survived:** Survival indicator (0 = No, 1 = Yes).
- **Pclass:** Ticket class (1st, 2nd, 3rd).
- **Name:** Passenger name.
- **Sex:** Gender of the passenger.
- **Age:** Age of the passenger.
- **SibSp:** Number of siblings/spouses aboard the Titanic.
- **Parch:** Number of parents/children aboard the Titanic.
- **Ticket:** Ticket number.
- **Fare:** Passenger fare.
- **Cabin:** Cabin number.
- **Embarked:** Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

## MACHINE LEARNING MODELS 
 
the machine learning models that were used in the Project are:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Gradient Boosting Classifier

## Installation

To run the analysis locally, you'll need Python 3.x and the following Python libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

You can install the necessary packages using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter