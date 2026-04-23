# Thesis: Fair Data, Biased Outcomes, and the Hidden Discrimination of AI Hiring Systems

## Overview
This repository contains Python scripts developed for my Master's thesis on bias detection and fairness in machine learning models applied to hiring decisions.

The project investigates whether statistically fair datasets lead to fair model outcomes, and evaluates bias mitigation techniques.

## Repository Structure
- data_preprocessing.py – Cleans and prepares the dataset
- model_training.py – Trains ML models (Random Forest, XGBoost, Neural Network)
- fairness_evaluation.py – Computes fairness metrics (SPD, DI, EOD, AOD)
- mitigation.py – Applies bias mitigation techniques (Reweighing, Threshold Adjustment)
- shap_analysis.py – Generates SHAP explanations and visualisations

## Dataset
The dataset used is a recruitment dataset (sourced from Kaggle).  
It includes candidate features such as age, gender, and recruitment strategy.

Note: The dataset is not uploaded due to size/privacy constraints.

## How to Run
1. Install required packages:
   pip install -r requirements.txt

2. Run scripts in the following order:
   - data_preprocessing.py
   - model_training.py
   - fairness_evaluation.py
   - mitigation.py
   - shap_analysis.py

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- shap
- aif360

## Key Concepts
- Fairness Metrics: Statistical Parity Difference (SPD), Disparate Impact (DI), Equalised Odds Difference (EOD), Average Odds Difference (AOD)
- Explainability: SHAP (Shapley Additive Explanations)
- Bias Mitigation: Pre-processing (Reweighing), Post-processing (Threshold Adjustment)

## Notes
- Age is binarised into two groups: Young (20–34) and Old (35+)
- Gender and Age are treated as protected attributes
- Results may vary depending on random seed and train-test split

## Author
Andrew Lee  
MSc in Management, University of Mannheim
