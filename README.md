# Telecom Churn Analysis

## Objective

The goal of this project is to **predict customer churn** and provide **insights** to help a telecom company improve customer retention by focusing on high-value customers and identifying churn risks.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Models Used](#models-used)
4. [Key Insights](#key-insights)
5. [Business Recommendations](#business-recommendations)
6. [Installation](#installation)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [Technologies Used](#technologies-used)

---

## Project Overview

This data-driven project seeks to derive actionable insights to enhance customer retention by focusing on high-value customers and identifying churn risks. The primary concern is the **high customer churn rate**, which leads to significant revenue loss. By conducting in-depth analysis and predictive modeling, we aim to identify customers likely to churn and the reasons behind their decisions.

The analysis includes:
- Exploratory Data Analysis (EDA)
- Predictive modeling using various machine learning algorithms
- Feature importance analysis
- Business recommendations based on insights

---

## Dataset Description

The dataset contains information about 7,043 telecom customers and various attributes such as:
- **Demographics**: Gender, Age, Married, Dependents
- **Subscription Details**: Tenure, Phone/Internet Service, Contract, Payment Method
- **Service Usage**: Monthly charges, Total charges, Extra services (Streaming TV, Streaming Music)
- **Churn Information**: Customer status (Stayed, Churned, Joined), Churn reasons

### Key Columns:
- **Customer Status**: Whether the customer has churned, stayed, or joined.
- **Churn Reason**: Reasons for churn, available for churned customers.
- **Monthly Charges & Total Charges**: Billing details to understand the financial impact.

---

## Models Used

Several machine learning models were used to predict customer churn and understand feature importance:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**
4. **K-Nearest Neighbors (KNN)**
5. **Gradient Boosting Classifier**
6. **XGBoost**
7. **Naive Bayes**
8. **Decision Tree Classifier**
9. **Deep Learning (Neural Network)**

Each model’s performance was evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

---

## Key Insights

- **Churned customers** tend to have **lower tenure** and are more likely to be on **month-to-month contracts**.
- **Offer E** was linked to a **higher churn rate**, suggesting it might not be aligned with customer expectations.
- Customers subscribing to **additional internet services** (e.g., online security, backup) are less likely to churn.
- **Fiber optic internet users** have a higher churn likelihood despite its popularity, indicating a need for improvement in service quality.

---

## Business Recommendations

- **Focus on High-Value Customers**: Implement loyalty programs for top spenders to increase retention.
- **Improve Device and Service Offers**: Competitors' better devices and offers are a common reason for churn, so enhance competitive offerings.
- **Promote Long-Term Subscriptions**: Encourage month-to-month users to shift to long-term contracts.
- **Upsell Internet and Streaming Services**: Customers who subscribe to additional services are more likely to stay.
- **Reevaluate Offer E**: Investigate why this offer leads to higher churn and consider modifications.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bhutto17/Telecom-Churn-Analysis.git
   cd Telecom-Churn-Analysis


2. Install the required packages:

   pip install -r requirements.txt


## How to Run

1. Load the Jupyter Notebook:
  
   jupyter notebook Telecom_Churn_Analysis.ipynb

2. Run each cell step by step to load the data, train models, and visualize the results.





## **Results**

The best-performing models for predicting churn were:
- **Random Forest** with an accuracy of 86.4%, precision of 91%, and recall of 89%.
- **XGBoost** with an accuracy of 85.3%, precision of 91.6%, and recall of 88.3%.
- **Deep Learning** with an accuracy of 85.3%, precision of 92.5%, and recall of 85.8%.

These models provide reliable predictions and can help the business focus on customers most at risk of churning.


## Technologies Used

- **Python**: For data manipulation and model building
- **Pandas**: Data manipulation and analysis
- **Seaborn & Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning models
- **XGBoost**: Gradient boosting model
- **TensorFlow/Keras**: Deep learning model
- **Jupyter Notebook**: For interactive data analysis and model building

