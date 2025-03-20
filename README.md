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

# Tools & Technologies 

We used Python to explore, clean, analyze and visualize our data. SHAP (SHapley Additive exPlanations) was used to determine the most important features for predicting the dermatological conditions.

To develop models that can be used to classify the data and predict the dermatological conditions, we used the following methods:

ExtraTrees
RandomForest
SVM
KNN
XGBoost
NaiveBayes
Decision Tree

## Installation Instructions 
To run the models and find the most important features, run the following notebooks in the 'Code' folder in the order listed below: 
* Preprocessing_&_Model_Selection.ipynb
* model.ipynb

To run the notebooks, you will need one of the following: 
* Python 
* Anaconda 
* Google Colab 

### Python 
1.  Download Python from python.org
2. During installation, check "Add Python to PATH" and complete the installation
3. Install Jupyter Notebook via command line (Terminal/Command Prompt): Enter 'pip install notebook'
4. Launch and Use Jupyter Notebook
5. Locate and run the notebooks 

### Anaconda 
1. Install Anaconda by downloading it from anaconda.com
2. Once installed, launch the Anaconda Navigator.
3. Click on the Jupyter Notebook tile 
4. Locate the files and select 'Run' to run the notebooks 

### Google Colab 
1. Go to colab.research.google.com
2. Sign in with your Google account
3. Upload the notebook 
4. Click 'Run All' under 'Runtime' to run the notebooks 
