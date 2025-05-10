# Heart Attack Prediction - Big Data Project

## Overview

This repository contains a scalable, end-to-end pipeline for predicting heart attack risk using a large, real-world US health dataset. We leverage PySpark for distributed data processing and Amazon S3 for cloud storage. Multiple machine learning models (Logistic Regression, Random Forest, and ANN) are built, evaluated, and deployed. The final solution is integrated into a user-friendly Flask web application for real-time risk prediction.

---

## High-Level Code Logic

1. **Data Ingestion & Quality Assurance**
   - Load the dataset from Amazon S3 or local disk using PySpark.
   - Perform data quality checks to ensure no missing values or inconsistencies.

2. **Exploratory Data Analysis (EDA)**
   - Conduct EDA in Jupyter notebooks (`Heart_Attack_Data_EDA_full_data.ipynb`, `Heart_Attack_Data_EDA_.ipynb`) to visualize feature distributions, trends, and class balance.

3. **Feature Engineering & Preprocessing**
   - Identify categorical and numerical features.
   - Apply StringIndexer and OneHotEncoder to categorical features.
   - Standardize numerical features.
   - Assemble all features into a single vector using PySpark's `VectorAssembler`.
   - Automate all preprocessing steps using a PySpark Pipeline for consistency and reproducibility.

4. **Model Training & Evaluation**
   - Split the data into training and testing sets (80/20).
   - Train multiple models (Logistic Regression, Random Forest, ANN) using PySpark MLlib.
   - Evaluate models on the test set; all models achieved ~80% accuracy.
   - Save the trained models and preprocessing pipeline in the `saved_models/` directory.

5. **Web Application Deployment**
   - The `app.py` file implements a Flask web server.
   - The web app loads the saved model and pipeline, processes user input, and returns real-time predictions.
   - The front end (in `templates/`) allows users to enter 31 health and lifestyle factors and view their risk prediction.

---

## Project Highlights

- **Scalable Big Data Pipeline:** Built using PySpark and Amazon S3 for efficient distributed processing.
- **Comprehensive EDA:** Explored trends, feature distributions, and class differences across demographics and health factors.
- **Robust Modeling:** Trained and evaluated multiple ML models with automated preprocessing and feature engineering.
- **Interactive Web App:** Real-time risk prediction via a user-friendly Flask interface.
- **Reproducibility:** Modular codebase and saved pipelines/models ensure easy extension and collaboration.

---

## Disclaimer

This tool is for demonstration and educational purposes only. It is not intended for real medical diagnosis or clinical use.

