# Multiple Disease Prediction System Using Machine Learning

An intelligent healthcare prediction system that uses Machine Learning algorithms to predict six different diseases through a unified Streamlit web application.

## 📌 Project Overview

The Multiple Disease Prediction System is a machine learning-based healthcare application designed to predict the likelihood of multiple diseases using patient-provided medical parameters.

The system integrates different machine learning algorithms into a single platform, allowing users to select a disease, enter relevant medical information, and receive prediction results through an interactive web interface.

The application supports six disease prediction modules:

- Diabetes
- Heart Disease
- Parkinson's Disease
- Stroke
- Liver Disease
- Chronic Kidney Disease

## 🎯 Project Objectives

- Develop a unified platform for multiple disease prediction.
- Integrate different machine learning classification algorithms.
- Provide real-time predictions through a web interface.
- Improve accessibility to machine learning-based healthcare tools.
- Compare the performance of different machine learning algorithms.
- Explore the application of AI and Machine Learning in healthcare.

## ✨ Key Features

- Multi-disease prediction through a single platform.
- Interactive user interface developed using Streamlit.
- Integration of multiple trained machine learning models.
- Medical parameter-based predictions.
- Data preprocessing and feature scaling.
- Comparative analysis of machine learning algorithms.
- Modular architecture for future improvements.

## 🧠 Machine Learning Algorithms

The research evaluates three machine learning algorithms:

### 1. Logistic Regression
A supervised machine learning algorithm used for classification problems. It provides a simple and interpretable baseline for disease prediction.

### 2. Support Vector Machine (SVM)
A classification algorithm that identifies optimal decision boundaries and performs effectively in high-dimensional feature spaces.

### 3. Random Forest
An ensemble learning algorithm that combines multiple decision trees to improve prediction performance and handle complex relationships in medical datasets.

**Research Finding:** Random Forest demonstrated the strongest overall performance among the evaluated algorithms in the reported experiments.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core Programming Language |
| Streamlit | Web Application Framework |
| Scikit-learn | Machine Learning |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| StandardScaler | Feature Scaling |

## 🏗️ System Workflow

1. User selects the disease to predict.
2. User enters the required medical parameters.
3. Input data undergoes preprocessing.
4. The corresponding trained machine learning model processes the input.
5. The system generates and displays the prediction result.

## 📊 Dataset and Preprocessing

The research uses publicly available medical datasets containing disease-specific clinical parameters.

Examples of input features include:

- Age
- Blood pressure
- Glucose levels
- Cholesterol levels
- Other disease-specific medical parameters

Data preprocessing includes:

- Data cleaning
- Missing value handling
- Feature transformation
- Feature scaling using StandardScaler
- Training and testing dataset splitting

## 📈 Results and Analysis

The research compares the performance of three machine learning algorithms using standard evaluation metrics:

- Accuracy
- Precision
- Recall
- Confusion Matrix

The experimental analysis identifies Random Forest as the best-performing approach among the evaluated algorithms.

## 📄 Research Paper

**Research Title:** Multiple Disease Prediction System Using Machine Learning

**Institution:** Sushant University, Gurugram, Haryana, India

**Department:** Computer Science and Engineering

**Research Authors:**

- Abhishek Kumar
- Shivam Mehta
- Vinayak Sarkar
- Dr. Shiksha Kumari

**Research Paper:** [Read the Complete Research Paper](Research_Paper/Multiple_Disease_Prediction_Research_Paper.pdf)

## 📁 Project Structure

```text
multiple-disease-prediction-using-machine-learning/
│
├── Research_Paper/
│   └── Multiple_Disease_Prediction_Research_Paper.pdf
│
├── models/
├── datasets/
├── assets/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
