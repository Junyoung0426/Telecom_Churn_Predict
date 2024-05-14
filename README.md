# Telecom_Churn_Predict
# Telecom Customer Churn Analysis

This repository contains a Jupyter Notebook that analyzes telecom customer churn data. The goal is to understand the factors that influence customer churn and to build predictive models to identify customers at risk of leaving.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
    - [Logistic Regression](#logistic-regression)
    - [Random Forest](#random-forest)
    - [Gradient Boosting](#gradient-boosting)
6. [Model Evaluation](#model-evaluation)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction
This project aims to analyze customer churn in the telecom industry. By understanding the key factors that influence churn, telecom companies can take proactive measures to retain customers. The notebook includes steps for data preprocessing, exploratory data analysis, modeling, and evaluation.

## Data Description
The dataset used in this analysis contains information about telecom customers, including their demographics, account information, and usage patterns. Key features include:

- `CustomerID`: Unique identifier for each customer
- `Gender`: Gender of the customer
- `SeniorCitizen`: Whether the customer is a senior citizen
- `Partner`: Whether the customer has a partner
- `Dependents`: Whether the customer has dependents
- `Tenure`: Number of months the customer has stayed with the company
- `PhoneService`: Whether the customer has phone service
- `MultipleLines`: Whether the customer has multiple lines
- `InternetService`: Type of internet service
- `OnlineSecurity`: Whether the customer has online security
- `OnlineBackup`: Whether the customer has online backup
- `DeviceProtection`: Whether the customer has device protection
- `TechSupport`: Whether the customer has tech support
- `StreamingTV`: Whether the customer has streaming TV
- `StreamingMovies`: Whether the customer has streaming movies
- `Contract`: Type of contract
- `PaperlessBilling`: Whether the customer has paperless billing
- `PaymentMethod`: Payment method used by the customer
- `MonthlyCharges`: Monthly charges
- `TotalCharges`: Total charges
- `Churn`: Whether the customer has churned

## Exploratory Data Analysis (EDA)
In the EDA section, we explore the dataset to understand the distribution of variables, identify missing values, and uncover relationships between features and the target variable (`Churn`). Visualizations and statistical summaries are used to gain insights.

## Data Preprocessing
Data preprocessing involves cleaning the dataset and preparing it for modeling. Steps include:

- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Splitting the data into training and testing sets

## Modeling
We build several machine learning models to predict customer churn:

### Logistic Regression
A simple yet effective model for binary classification tasks. It estimates the probability of a customer churning based on the input features.

### Random Forest
An ensemble learning method that combines multiple decision trees to improve predictive performance and control overfitting.

### Gradient Boosting
Another ensemble technique that builds models sequentially, with each new model correcting errors made by previous ones.

## Model Evaluation
Model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. We compare the results of different models to select the best-performing one.

## Conclusion
We summarize the findings of the analysis and provide recommendations for telecom companies to reduce customer churn.

## References
- [Telecom Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- [Machine Learning Documentation](https://scikit-learn.org/stable/user_guide.html)
