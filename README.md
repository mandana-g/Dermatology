# Project Goal

Determine what clinical and histopathological features are the most significant predictors that classify patients into specific erythemato-squamous diseases.

# Objectives

1. Analyzing and investigating the dataset to find some insights into our classification model.
2. Training different classification models to come up with the best possible classifications in terms of predefined metrics.
3. Finding the important features based on the detection outcome of our best finding classification model.

# Methodology

1. Exploratory Data Analysis (EDA):
   1.1. Analyze the distribution of clinical and histopathological features.
   1.2. Investigate correlations between features and identify relationships that could aid in classification.
   1.3. Visualize data distributions and correlations to gain insights into the dataset's structure.

2. Data Preprocessing:
   Clean the dataset by:
   2.1. Handling Missing Values:
    The dataset contains 8 missing values in age instances. We addressed this by two different approaches to ensure the integrity of the dataset:
       2.1.1. Removing rows with missing values in 'age' 
       2.1.2. Imputing missing 'age' values with median

   2.2. Feature Transformation:
   We applied a log transformation to skewed features (e.g., 'koebner phenomenon', 'polygonal papules') to reduce skewness and improve model performance by making the data more normally distributed.

   2.3 Standardization:
   Numerical features were standardized using StandardScaler to ensure all features contribute equally to the model by scaling them to have a mean of 0 and standard deviation of 1.

3. Model Development and Evaluation:
   3.1. Develop multiple machine learning models, such as Random Forest, SVM, Decision Trees, and XGBoost, to classify the diseases.
   3.2. Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

4. Hyperparameter Tuning:

   Fine-tune the model hyperparameters using techniques like Grid Search and Cross-validation to optimize performance and avoid overfitting.

5. Feature Importance Analysis:

   Analyze the importance of different features in making predictions, providing insights into which clinical and histopathological factors are most significant in diagnosing erythemato-squamous diseases.

6. Model Deployment (Optional):

   Build a user interface and/or API for deploying the trained model, allowing easy access for dermatologists or researchers to use the model for predictions.

# Task Breakdown

| No. | Task | Contributor |
|---- | -----|-------------|
| 1   |EDA(Exploratory Data Analysis)| all team members |
| 2   | Data preprocessing( standardization, imputation, encoding(?)) | all team members |
| 3   | Model Development( suggested models: Random Forest, XGBoost, Neural Network, SVM, KNN, Decision Tree) | each team member separately |
| 4   | Initial Model Evaluation | same as above |
| 5   | Hyper Parameter Tuning (Grid Search, Cross-validation, etc)| same as above |
| 6   | Final Model Evaluation | same as above |
| 7   | Feature Importance Analysis | same as above |
| 8   | Presentation/Explanation    | same as above | 
| 9   | (Optional) User Interface, APIs | TBD |
| 10  | (Future Works): 1. Designing semi-supervised learning by use of Generative Adversarial Network. 2. Optimization  | TBD |

# Dataset Information

This database contains 34 attributes, 33 of which are linear valued and one of them is nominal. 

The differential diagnosis of erythemato-squamous diseases is a real problem in dermatology. They all share the clinical features of erythema and scaling, with very little differences. The diseases in this group are psoriasis, seboreic dermatitis, lichen planus, pityriasis rosea, cronic dermatitis, and pityriasis rubra pilaris. Usually a biopsy is necessary for the diagnosis but unfortunately these diseases share many histopathological features as well. Another difficulty for the differential diagnosis is that a disease may show the features of another disease at the beginning stage and may have the characteristic features at the following stages. Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features. The values of the histopathological features are determined by an analysis of the samples under a microscope. 

In the dataset constructed for this domain, the family history feature has the value 1 if any of these diseases has been observed in the family, and 0 otherwise. The age feature simply represents the age of the patient. Every other feature (clinical and histopathological) was given a degree in the range of 0 to 3. Here, 0 indicates that the feature was not present, 3 indicates the largest amount possible, and 1, 2 indicate the relative intermediate values.

The names and id numbers of the patients were recently removed from the database.

Reference: https://archive.ics.uci.edu/dataset/33/dermatology

# Tools & Technologies 
We used Python to explore, clean, analyze and visualize our data. 

## Libraries 
The following libraries were used: 
* ucimlrepo
* optuna
* shap
* lime
* pandas
* numpy 
* matplotlib
* seaborn 
* sklearn
* xgboost 
* joblib
* warnings
* random
* os

## Installation Instructions 
To run the models and find the most important features, run the notebooks in the 'Code' folder 

To run the notebooks, you will need one of the following: 
* IDE
* Google Colab 

### IDE 
1.  Download an IDE onto your computer e.g. Python, VSCode
2.  Launch the IDE 
3. Ensure that you have the appropriate libaries installed 
4. Locate and run the notebooks

### Google Colab 
1. Go to colab.research.google.com
2. Sign in with your Google account
3. Upload the notebook 
4. Click 'Run All' under 'Runtime' to run the notebooks 

## Models
To develop models that can be used to classify the data and predict the dermatological conditions, we used the following methods:
* NaiveBayes 
* KNN
* SVM
* Decision Tree
* XGBoost
* Random Forest 
* Extra Trees 
