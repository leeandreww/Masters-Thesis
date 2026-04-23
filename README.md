# Fairness Audit of Recruitment Models

This repository contains Python scripts used to audit bias, evaluate model performance, and compare mitigation strategies on a recruitment dataset.

## What This Project Does

- Produces Random Forest feature-importance analysis
- Audits raw dataset bias before modeling (Gender and Age groups)
- Trains and evaluates three model families:
  - Neural Network
  - Random Forest
  - XGBoost
- Compares baseline vs mitigation approaches:
  - Re-weighting
  - Group-specific threshold adjustment
- Produces SHAP-based feature attribution tables


## Repository Structure

- recruitment_data.csv
- raw_data_audit/
  - raw_bias_audit.py
  - raw_bias_audit.csv
- neural_network/
  - neural_network_audit.py
  - nn_performance.csv
  - nn_fairness.csv
- random_forest/
  - random_forest_audit.py
  - rf_performance.csv
  - rf_fairness.csv
- xgboost/
  - xgboost_audit.py
  - xgb_performance.csv
  - xgb_fairness.csv
- shap_analysis/
  - shap_analysis.py
  - table_nn_shap.csv
  - table_rf_shap.csv
  - table_xgb_shap.csv
- rf_importance/
  - rf_importance.py
  - rf_feature_importance.png


## Requirements

- Python 3.10+ (the project was tested in a virtual environment)
- The dataset file recruitment_data.csv available in:
  - project root for most scripts
  - each subfolder script can also use a local copy if coded that way

Main Python libraries used:
- pandas
- numpy
- scikit-learn
- aif360
- shap
- matplotlib
- joblib

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

Suggested commands on macOS/Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install pandas numpy scikit-learn aif360 shap matplotlib joblib

## Recommended Run Order

Run from the project root directory:

    python rf_importance/rf_importance.py
    python raw_data_audit/raw_bias_audit.py
    python neural_network/neural_network_audit.py
    python random_forest/random_forest_audit.py
    python xgboost/xgboost_audit.py
    python shap_analysis/shap_analysis.py
    

## Outputs

- Bias and performance CSV files are written into each script folder.

