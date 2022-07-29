# Introduction 

This is a Jupyter Notebook for a Classification problem using Scikit-Learn. <br>

The objective of this model is to correctly predict which Credit Card Customers will default on their Credit Card Bill. The dataset originates from a Taiwanese Financial Institution who provides credit card to their customers. From the dataset, we have certain demographic information about the customers, credit card bills, how much credit card bill they paid and whether they have defaulted on their credit card bill. <br><br>

### Column Names and Descriptions

Below provides more information on the columns and what they signify:

| Column Name | Description | 
|---| --- |
| ID | Automatically generated ID |
|LIMIT_BAL                 |Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit                                                     |
|SEX	                      |Values: 1 = male; 2 = female                                                     |
|EDUCATION                 |Values: 1 = graduate school; 2 = university; 3 = high school; 4 = others                                                     |
|MARRIAGE                  |Marital status. Values: 1 = married; 2 = single; 3 = others                                                     |
|AGE	                      |Age in Years                                                     |
|PAY_0 to PAY_6            |History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: <br> X6 = the repayment status in September, 2005; <br> X7 = the repayment status in August, 2005; . . .; <br> X11 = the repayment status in April, 2005. <br><br> The measurement scale for the repayment status is: <br> -1 = pay duly; <br> 1 = payment delay for one month; <br> 2 = payment delay for two months;. . .; <br> 8 = payment delay for eight months; <br> 9 = payment delay for nine months and above. <br> PAY_n is sorted in descending order, in which PAY_0 is  September 2005 and PAY_6 April 2005. |
|BILL_AMT1 to BILL_AMT6    |Amount of bill statement (NT dollar). <br> BILL_AMT1 =  amount of bill statement in September 2005; <br> BILL_AMT2 = amount of bill statement in August, 2005; . . .; <br> BILL_AMT6 = amount of bill statement in April, 2005.|
|PAY_AMT1 to PAY_AMT6      |Amount of previous payment (NT dollar). <br> PAY_AMT1 = amount paid in September, 2005; <br> PAY_AMT2 = amount paid in August, 2005; . . .; <br> PAY_AMT6 = amount paid in April, 2005. |
|default payment next month| Probability of Credit Card bill default the following month. <br> Values: 1 = Default, 0= Non-default |


Source: <a href="https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients">Link here</a>

# Summary of the Notebook

I performed the following steps for this notebook:

1. Performed Exploratory Data Analysis (EDA)
2. Dropped ```Sex, Education, Marriage, Age``` columns
3. Prepared models, looking at Accuracy Metric
4. Using the best model, perform Hyperparameter Tuning using GridCVSearch 

## Model Preparation and Building

For Model Preparation, I will be performing a Stratified Shuffle Split to keep the Classifier proportion across Train and Test Dataset

For this first iteration, I will be building a simple Logistic Regression.

I have chosen the following Models for Training:
1. Logistic Regression
2. Logistic Regression with Cross Validation
3. Decision Tree Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier
6. Support Vector Machine

At this stage, the default settings have been kept as much as possible for this stage. If any changes, only the max number of iterations are changed.
After training, I will look at <b>Accuracy metric</b> to see the best model to use for hyperparameter tuning 


# Reproducing this Notebook

The ```requirements.txt``` is availabe in this repository. It can be run either using Anaconda or Pip.

Should the ```requirements.txt``` not work, please see Alternative Installation Method

### Alternative Installation Method

This notebook requires the following Python version and modules available. Below are the instructions to create using > Anaconda Prompt 

<b> Installation Steps for Python and relevant Libraries (Anaconda)</b>
1. Install Anaconda Distribution (link: https://www.anaconda.com/products/distribution)
2. After installation, open Anaconda Prompt
3. Assuming no environment created, create one using the following command: ``` conda create --name ML python=3.9.7 numpy pandas scikit-learn matplotlib notebook ```
4. Once environment created, use command ```conda activate ML```

Please install module 'xlrd' before running the code. XLRD is needed to enable Pandas to read Excel files (.xls file types). Please see installation steps using either <b>PIP</b> or <b>CONDA</b> below:

<b> Installation Steps (Pip)</b>
1. Open Windows Command Prompt
2. At Command Prompt, type ```pip install xlrd```
3. Proceed to run below

<b> Installation Steps (Anaconda Prompt)</b>
1. Open Anaconda Prompt
2. Change to environment by ```conda activate ML```
3. At Command Prompt, type ```conda install xlrd```

<b>After installing ```xlrd```, please perform following steps: </b>

1. At Anaconda Prompt, type: ```conda activate ML```
2. Type: ```jupyter notebook```
3. Open this Jupyter Notebook file (.ipynb) at Jupyter Notebook
4. Proceed to run below


# Observation and Conclusion

1. Even after changing the hyperparameter tuning using GridSearchCV, the accuracy doesn't increase beyond 81%
2. Looking at the Confusion Matrix, there are less False Positive cases for the Random Forest Classifier that went through hyperparameter tuning.

# Concluding Notes:

1. Looking at this results and back to the academic paper written together with the Credit Card Default Data, this is the highest accuracy we can achieve without using a Neural Network Architecture
2. This is also in line with their results, that a neural network architecture can achieve the highest accuracy (or R squared value of 0.99).
3. Also, there is a danger of Random Forest Classifier being a "black box" (meaning we may not know what is happening underneath), so explainability may be lesser, as compared to using a Logistic Regression


# Reference

Citation for original Academic paper
```bibtex
@article{yeh2009comparisons,
  title={The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients},
  author={Yeh, I-Cheng and Lien, Che-hui},
  journal={Expert systems with applications},
  volume={36},
  number={2},
  pages={2473--2480},
  year={2009},
  publisher={Elsevier}
}
```

The dataset was obtained from UCI Machine Learning Repository <a href="https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients"> Link here </a>.