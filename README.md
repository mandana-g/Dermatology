# Project Goal
Determine what clinical and histopathological features are the most significant predictors that classify patients into specific erythemato-squamous diseases.

# Task Breakdown
 1. EDA(Explaratory Data Analysis): Contributors(Done by each of the team members separately)
 2. Data preprocessing (strandardization, imputation, encoding(?)):  Contributors(Done by each of the team members separately)
 3. Model Development: (suggested models: Random Forest, XGBoost, Neural Network, SVM, KNN, Decision Tree) contributors(Each team member will work on a different model)
 4. Hyper Parameter Tuning including Grid Search, Cross-validation: contributors(same as above)
 5. Model Evaluation: contributors(same as above)
 6. Feture Importrance Analysis: contributors(same as above)
 7. Presentation/Explanation: contributors(same as above)
 8. (Optional) User Interface, APIs: contributors(TBD)
 9. (Future Works): 1. Designing semi-supervised learning by use of Generative Adverserial Network. 2. Optimization: contributors(TBD)

# Dataset Information

This database contains 34 attributes, 33 of which are linear valued and one of them is nominal. 

The differential diagnosis of erythemato-squamous diseases is a real problem in dermatology. They all share the clinical features of erythema and scaling, with very little differences. The diseases in this group are psoriasis, seboreic dermatitis, lichen planus, pityriasis rosea, cronic dermatitis, and pityriasis rubra pilaris. Usually a biopsy is necessary for the diagnosis but unfortunately these diseases share many histopathological features as well. Another difficulty for the differential diagnosis is that a disease may show the features of another disease at the beginning stage and may have the characteristic features at the following stages. Patients were first evaluated clinically with 12 features. Afterwards, skin samples were taken for the evaluation of 22 histopathological features. The values of the histopathological features are determined by an analysis of the samples under a microscope. 

In the dataset constructed for this domain, the family history feature has the value 1 if any of these diseases has been observed in the family, and 0 otherwise. The age feature simply represents the age of the patient. Every other feature (clinical and histopathological) was given a degree in the range of 0 to 3. Here, 0 indicates that the feature was not present, 3 indicates the largest amount possible, and 1, 2 indicate the relative intermediate values.

The names and id numbers of the patients were recently removed from the database.

Reference: https://archive.ics.uci.edu/dataset/33/dermatology
