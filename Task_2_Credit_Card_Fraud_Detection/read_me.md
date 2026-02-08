# 💳 Credit Card Fraud Detection

### CODSOFT Machine Learning Internship – Task 2

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques.
The goal is to build a model that can classify transactions as **fraudulent or legitimate** based on transaction features.

Credit card fraud detection is a real-world problem faced by financial institutions, where accurate prediction helps prevent financial loss and enhances security.

This project was completed as part of the **CODSOFT Machine Learning Internship**.

---

## 🎯 Objective

To develop a machine learning model that:

* Identifies fraudulent transactions
* Handles imbalanced datasets
* Evaluates model performance using proper metrics
* Improves fraud detection accuracy

---

## 📊 Dataset Description

The dataset contains transaction-related features such as:

* Transaction amount
* Customer and merchant details
* Transaction time and category
* Fraud label (target variable)

### Target Variable:

`is_fraud`

* 0 → Legitimate transaction
* 1 → Fraudulent transaction

The dataset is **highly imbalanced**, meaning fraud cases are very few compared to legitimate transactions.

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

---

## 🧠 Machine Learning Concepts Used

* Data preprocessing
* Handling imbalanced dataset
* Feature scaling
* Binary classification
* Model evaluation metrics

---

## 🔄 Project Workflow

### 1️⃣ Data Loading

The dataset was loaded using pandas and explored to understand its structure and features.

### 2️⃣ Data Preprocessing

* Removed unnecessary columns
* Converted categorical data into numerical form
* Scaled numerical features
* Prepared dataset for training

### 3️⃣ Handling Imbalanced Data

Since fraud cases are rare, the dataset is imbalanced.
Therefore, evaluation was not based on accuracy alone but also on precision, recall, and F1-score.

### 4️⃣ Train-Test Split

Dataset was divided into:

* Training set (for model learning)
* Testing set (for performance evaluation)

### 5️⃣ Model Building

Three machine learning models were applied:

* Logistic Regression
* Decision Tree
* Random Forest (best performing model)

### 6️⃣ Model Evaluation

Models were evaluated using:

* Accuracy score
* Precision
* Recall
* F1-score
* Classification report

Random Forest provided the best performance in detecting fraudulent transactions.

---

## 📈 Result

The machine learning model successfully detects fraudulent transactions with strong evaluation metrics.
This project demonstrates how ML can be applied in real-world financial fraud detection systems.

---

## 📚 Learning Outcomes

Through this project, I learned:

* Handling real-world imbalanced datasets
* Building binary classification models
* Evaluating models using proper metrics
* Understanding fraud detection systems
* Applying machine learning in finance domain

---

## 🚀 Future Improvements

* Apply advanced algorithms (XGBoost, Gradient Boosting)
* Perform hyperparameter tuning
* Deploy model as web application
* Improve fraud detection accuracy

---

## 👩‍💻 Author

**Komal Bhogale**
Machine Learning Intern – CODSOFT
Aspiring Data Scientist passionate about data and analytics.

---

## 🏷️ Tags

`#MachineLearning` `#DataScience` `#FraudDetection`
`#CODSOFT` `#Internship` `#Python`
