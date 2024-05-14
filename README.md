# Telecom Customer Churn Analysis

This repository contains a Jupyter Notebook that analyzes telecom customer churn data. The goal is to understand the factors that influence customer churn and to build predictive models to identify customers at risk of leaving.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
    - [Logistic Regression](#logistic-regression)
    - [Decision Tree](#decision-tree)
    - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
    - [Support Vector Machine (SVM)](#support-vector-machine-svm)
    - [Naive Bayes](#naive-bayes)
    - [XGBoost](#xgboost)
    - [Gradient Boosting](#gradient-boosting)
    - [LightGBM](#lightgbm)
    - [CatBoost](#catboost)
6. [Model Evaluation & Hyper Parameter](#model-evaluation-&-Hyper-Parameter)
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

### Decision Tree
A non-parametric supervised learning method used for classification. It splits the data into subsets based on the most significant differentiators in the input features.

### K-Nearest Neighbors (KNN)
A non-parametric method used for classification. It classifies a data point based on how its neighbors are classified.

### Support Vector Machine (SVM)
A supervised learning model used for classification by finding the hyperplane that best separates the classes in the feature space.

### Naive Bayes
A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.

### XGBoost
An optimized gradient boosting framework designed for speed and performance. It builds models sequentially to correct errors of previous models.

### Gradient Boosting
An ensemble technique that builds models sequentially, with each new model correcting errors made by previous ones.

### LightGBM
A gradient boosting framework that uses tree-based learning algorithms. It is designed for distributed and efficient training of large datasets.

### CatBoost
A gradient boosting algorithm that handles categorical features naturally without the need for extensive preprocessing.

## Model Evaluation & Hyper Parameter
Model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. We compare the results of different models to select the best-performing one.

## Conclusion
We summarize the findings of the analysis and provide recommendations for telecom companies to reduce customer churn.

## References
- [Telecom Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- [Machine Learning Documentation](https://scikit-learn.org/stable/user_guide.html)
