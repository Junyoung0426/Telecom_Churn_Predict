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
6. [Model Evaluation](#model-evaluation)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction
This project aims to analyze customer churn in the telecom industry. By understanding the key factors that influence churn, telecom companies can take proactive measures to retain customers. The notebook includes steps for data preprocessing, exploratory data analysis, modeling, and evaluation.

## Data Description
The dataset used in this analysis contains information about telecom customers, including their demographics, account information, and usage patterns. The table below provides details about each column in the dataset.

| Column Name                 | Data Type | Description                              |
|-----------------------------|-----------|------------------------------------------|
| `CustomerID`                | object    | 고객의 고유 식별 번호                    |
| `Gender`                    | object    | 고객의 성별                              |
| `Age`                       | int64     | 고객의 나이                              |
| `Married`                   | object    | 결혼 여부 (Yes, No)                      |
| `Number of Dependents`      | int64     | 부양 가족의 수                           |
| `City`                      | object    | 고객의 거주 도시                        |
| `Zip Code`                  | int64     | 고객의 우편 번호                         |
| `Latitude`                  | float64   | 고객 거주지의 위도                      |
| `Longitude`                 | float64   | 고객 거주지의 경도                      |
| `Number of Referrals`       | int64     | 고객이 추천한 다른 고객 수              |
| `Tenure in Months`          | int64     | 서비스 이용 개월 수                     |
| `Offer`                     | object    | 고객이 받은 프로모션 형태               |
| `Phone Service`             | object    | 전화 서비스 가입 여부 (Yes, No)          |
| `Multiple Lines`            | object    | 여러 회선 사용 여부 (Yes, No, No phone service) |
| `Internet Service`          | object    | 인터넷 서비스 가입 여부 (Yes, No)        |
| `Internet Type`             | object    | 인터넷 서비스 종류 (DSL, Fiber optic, None) |
| `Online Security`           | object    | 온라인 보안 서비스 사용 여부             |
| `Online Backup`             | object    | 온라인 백업 서비스 사용 여부             |
| `Device Protection Plan`    | object    | 장비 보호 계획 가입 여부                 |
| `Premium Tech Support`      | object    | 프리미엄 기술 지원 서비스 사용 여부      |
| `Streaming TV`              | object    | TV 스트리밍 서비스 사용 여부             |
| `Streaming Movies`          | object    | 영화 스트리밍 서비스 사용 여부           |
| `Streaming Music`           | object    | 음악 스트리밍 서비스 사용 여부           |
| `Unlimited Data`            | object    | 무제한 데이터 서비스 사용 여부           |
| `Contract`                  | object    | 계약 유형 (Month-to-month, One year, Two year) |
| `Paperless Billing`         | object    | 종이 없는 청구서 사용 여부 (Yes, No)     |
| `Payment Method`            | object    | 결제 수단 (Electronic check, Mailed check 등) |
| `Monthly Charge`            | float64   | 매월 청구되는 요금                       |
| `Total Charges`             | float64   | 가입 후 총 청구된 금액                   |
| `Total Refunds`             | float64   | 고객에게 반환된 총 금액                  |
| `Total Extra Data Charges`  | int64     | 추가 데이터 사용에 따른 총 추가 요금     |
| `Total Long Distance Charges`| float64  | 장거리 통화에 대한 총 요금               |
| `Total Revenue`             | float64   | 총 수익 (모든 요금 포함)                 |
| `Customer Status`           | object    | 현재 고객 상태 (계속, 이탈 등)           |
| `Churn Category`            | object    | 고객 이탈 유형                           |
| `Churn Reason`              | object    | 고객 이탈의 구체적인 이유                |

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

## Model Evaluation
Model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. We compare the results of different models after Hyper Parameter Tuning to select the best-performing one.

## Conclusion
We summarize the findings of the analysis and provide recommendations for telecom companies to reduce customer churn.

## References
- [Telecom Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- [Machine Learning Documentation](https://scikit-learn.org/stable/user_guide.html)
