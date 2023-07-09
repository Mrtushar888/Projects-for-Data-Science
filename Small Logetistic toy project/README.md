
# Logistic Regression for Placement Prediction

This repository contains Python code that demonstrates the use of logistic regression for placement prediction. The code uses the scikit-learn library to build a logistic regression model and predict the placement outcome for students based on their CGPA (Cumulative Grade Point Average) and IQ scores.

## Dataset

The dataset used in this code is stored in the placement.csv file. It contains information about students, including their CGPA, IQ scores, and placement status (0 for not placed and 1 for placed). The code reads the dataset using pandas and performs data preprocessing steps.

## Dependencies

The code requires the following dependencies:

pandas
numpy
scikit-learn
matplotlib
mlxtend
You can install the dependencies using the pip install command:

pip install pandas numpy scikit-learn matplotlib mlxtend
## Usage

1) Ensure that you have the necessary dependencies installed.
2) Clone this repository or download the code files.
3) Place the placement.csv file in the same directory as the Python code.
4) Run the code using a Python interpreter or Jupyter Notebook.


## Steps

1) The code begins by importing the necessary libraries.
2) It reads the dataset from the placement.csv file using pandas.
3) Data preprocessing is performed, including removing unnecessary columns and handling missing values.
4) Data visualization is carried out using scatter plots to show the relationship between CGPA, IQ, and placement.
5) The dataset is split into training and testing sets using the train_test_split function from scikit-learn.
6) Feature scaling is applied using the StandardScaler to normalize the data.
7) The logistic regression model is created using the LogisticRegression class from scikit-learn.
8) The model is trained using the training data.
9) Predictions are made on the testing data using the trained model.
10) Accuracy of the model is calculated using the accuracy_score function from scikit-learn.
11) Decision regions are plotted using the plot_decision_regions function from mlxtend.
## Results

The code outputs the accuracy score of the logistic regression model and plots the decision regions for visualization. The accuracy score indicates how well the model predicts the placement outcome based on the CGPA and IQ scores.

Feel free to modify the code or dataset as per your requirements and experiment with different parameters to achieve better results.

Note: Ensure that you have the necessary data in the placement.csv file in the correct format before running the code.