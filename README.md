# Credit Card Fraud Detection

This repository contains a Jupyter Notebook (Project.ipynb) that demonstrates a credit card fraud detection project using data science techniques.

## Dataset

The project uses the [credit card fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. The dataset contains transactions made by credit cards in September 2013 by European cardholders. It consists of a labeled class indicating whether the transaction is fraudulent or not, along with various numerical features derived from the transaction.

## Project Description

In this project, we aim to build a machine learning model that can accurately detect fraudulent credit card transactions. The notebook provides step-by-step instructions on data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation. Here is an overview of the steps taken and their purposes:

1. **Data Preprocessing**: This step involves handling missing values, scaling numerical features, and encoding categorical variables. Preprocessing ensures that the data is in a suitable format for further analysis and modeling.

2. **Exploratory Data Analysis (EDA)**: EDA helps us gain insights into the dataset by visualizing and analyzing its various aspects. In this project, the following EDA techniques were applied:

   - **Correlation Heatmap**: A correlation heat map was created to understand the relationships between different variables in the dataset. This helped identify any significant correlations between features and the target variable (fraudulent transactions), aiding in feature selection and understanding potential predictors.

   - **Data Visualization**: Various visualizations, such as histograms, box plots, and scatter plots, were used to examine the distribution of features, identify outliers or anomalies, and understand the differences between fraudulent and non-fraudulent transactions.

3. **Model Training**: Model training involves building and training machine learning models on the preprocessed data. In this project, multiple types of models were built, including logistic regression, random forest, and support vector machines. The purpose of training multiple models is to compare their performance and determine which one provides the best results for credit card fraud detection.

4. **Feature Importance**: To gain insights into the significant features impacting the model's predictions, a feature importance graph was plotted. This graph helps identify the most influential features in distinguishing between fraudulent and non-fraudulent transactions.

5. **Cross-Validation**: Cross-validation is used for model evaluation in this project. Two types of cross-validation techniques were employed:

   - **Stratified K-Fold**: Stratified K-fold is used to ensure that each fold in the cross-validation process maintains the same class distribution as the original dataset. This technique helps mitigate the impact of class imbalance and provides a more robust evaluation of the models' performance.

   - **Repeated K-Fold**: Repeated K-fold is applied to repeat the cross-validation process multiple times. It helps reduce the variability in the evaluation results and provides a more reliable estimation of the models' performance.

6. **Model Evaluation**: Model evaluation helps us assess the performance of the trained models. In this project, the following evaluation metrics were used:

   - **Accuracy**: The accuracy of the models in correctly predicting fraudulent and non-fraudulent transactions.
   - **Precision**: The precision of the models in identifying true fraudulent transactions out of the total transactions predicted as fraudulent.
   - **Recall**: The recall of the models in capturing all actual fraudulent transactions out of the total fraudulent transactions present.
   - **F1 Score**: The F1 score, which is a harmonic mean of precision and recall, provides a balanced measure of model performance.

   Additionally, the evaluation includes determining the optimal threshold for classification and analyzing the models' performance using different threshold values.

7. **Results Storage**: The project stores various results to analyze and compare the models' performance. This includes:

   - **Confusion Matrix**: The confusion matrix helps visualize the performance of the models in terms of true positives, true negatives, false positives, and false negatives.
   - **ROC AUC Curve**: The ROC AUC curve is plotted to analyze the models' performance

## Installation

To run the notebook locally, you need to have Python installed, along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

You can install the required libraries by running the following command:

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

## Usage

1. Clone the repository:
git clone https://github.com/Sanchitg17/DataScience_Credit-Card-Fraud-Detection.git

2. Navigate to the project directory:
cd DataScience_Credit-Card-Fraud-Detection

3. Launch Jupyter Notebook:
jupyter notebook

4. Open the `Project.ipynb` notebook in Jupyter.

5. Follow the instructions provided in the notebook to preprocess the data, train the model, and evaluate its performance.

## Contribution

Contributions to this project are welcome. If you find any issues or want to enhance the project, feel free to submit a pull request.

