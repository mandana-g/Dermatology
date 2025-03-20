# Project Goal
Determine what clinical and histopathological features are the most significant predictors that classify patients into specific erythemato-squamous diseases.

# Task Breakdown

| No. | Task | Contributor |
|---- | -----|-------------|
| 1   |EDA(Explaratory Data Analysis)| all team members |
| 2   | Data preprocessing( strandardization, imputation, encoding(?)) | all team members |
| 3   | Model Development( suggested models: Random Forest, XGBoost, Neural Network, SVM, KNN, Decision Tree) | each team member separately |
| 4   | Initial Model Evaluation | same as above |
| 5   | Hyper Parameter Tuning (Grid Search, Cross-validation, etc)| same as above |
| 6   | Final Model Evaluation | same as above |
| 7   | Feture Importrance Analysis | same as above |
| 8   | Presentation/Explanation    | same as above | 
| 9   | (Optional) User Interface, APIs | TBD |
| 10  | (Future Works): 1. Designing semi-supervised learning by use of Generative Adverserial Network. 2. Optimization  | TBD |



# Dataset Information

This database contains 34 attributes, 33 of which are linear valued and one of them is nominal. 

The differential diagnosis of erythemato-squamous diseases is a real problem in dermatology. They all share the clinical features of erythema and scaling, with very little differences. The diseases in this group are psoriasis, seboreic dermatitis, lichen planus, pityriasis rosea, cronic dermatitis, and pityriasis rubra pilaris. Usually a biopsy is necessary for the diagnosis but unfortunately these diseases share many histopathological features as well. Another difficulty for the differential diagnosis is that a disease may show the features of another disease at the beginning stage and may have the characteristic features at the following stages. Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features. The values of the histopathological features are determined by an analysis of the samples under a microscope. 

In the dataset constructed for this domain, the family history feature has the value 1 if any of these diseases has been observed in the family, and 0 otherwise. The age feature simply represents the age of the patient. Every other feature (clinical and histopathological) was given a degree in the range of 0 to 3. Here, 0 indicates that the feature was not present, 3 indicates the largest amount possible, and 1, 2 indicate the relative intermediate values.

The names and id numbers of the patients were recently removed from the database.

Reference: https://archive.ics.uci.edu/dataset/33/dermatology


# Purpose & Overview

The purpose of this project is to build and evaluate machine learning models to classify patients into specific erythemato-squamous disease categories using clinical and histopathological data. The project also focuses on identifying the most significant features that influence these classifications to provide valuable insights for medical professionals.

**Business question:**  
*What clinical and histopathological features are the most significant predictors that classify patients into specific erythemato-squamous diseases?*

This project has two core objectives:
- **Develop and compare multiple classification models** to determine which achieves the highest predictive accuracy for diagnosing six types of erythemato-squamous diseases.
- **Identify key clinical and histopathological predictors** that contribute most to the classification decisions.

## How this is approached:

- Multiple machine learning models are evaluated, including **Random Forest, Extra Trees, SVM, KNN, XGBoost,** and **Naive Bayes**.
- A preprocessing pipeline is used to streamline **data imputation** (for missing values in "age") and model training.
- **Class imbalance** is addressed by applying `class_weight="balanced"` in applicable algorithms.
- **Optuna** is used for hyperparameter tuning, optimizing parameters such as `n_estimators`, `max_depth`, and `min_samples_split` for the selected models.
- To explain model behavior and highlight important features, several interpretability techniques are applied:
  - **Built-in feature importance** for tree-based models.
  - **Permutation importance** for models without native feature importance.
  - **LIME** for local-level explanations of individual predictions.
  - **SHAP** to provide a global understanding of how features influence model outputs across all classes.

The project delivers insights into both the **best-performing model** for this task and the **critical features** driving its decisions, offering actionable knowledge for potential clinical applications.
