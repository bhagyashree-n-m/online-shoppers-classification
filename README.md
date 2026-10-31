# Online Shoppers Purchase Intention Prediction Using Machine Learning

## A. Problem Statement

This is a Binary Classification problem. The objective of this project is to develop and evaluate machine learning classification models to predict whether an online shopping session will result in a purchase.

The project uses visitor behavior, browsing activity, session characteristics, and other attributes to classify shopping sessions into two categories: purchase and no purchase.


## B. Dataset Description

**Dataset Name:** Online Shoppers Purchase Intention

**Dataset Source:** [Kaggle — Online Shoppers Purchase Intention](https://www.kaggle.com/datasets/sairupeshl/online-shoppers-purchase-intention)

The dataset contains information about online shopping sessions, including visitor browsing behavior, page visits, time spent on pages, session characteristics, and visitor information.
The dataset consists of 10 numerical and 8 categorical attributes. The 'Revenue' attribute can be used as the class label.
shopping.csv contains 12,330 entries of user session data. Each row represents a different user's browsing session.

Key Columns:

- **Administrative, Informational, ProductRelated:** Number of pages visited in each category.
- **Duration:** Time spent in each category.
- **BounceRates, ExitRates:** Metrics measuring user engagement and site departure.
- **PageValues:** The average value of the page viewed before the user completed a transaction.
- **Month, Region, Browser:** Categorical data about the user.
- **Revenue:** The target label (TRUE = Purchase, FALSE = No Purchase).

The target variable is `Revenue`, which indicates whether a purchase occurred during a shopping session.

- `True` / `1`: Purchase occurred.
- `False` / `0`: No purchase occurred.

This is a binary classification problem. The dataset is divided into training and testing sets using an 80:20 split, with stratified sampling to preserve the target class distribution. Numerical features are imputed and standardized, while categorical features are imputed and one-hot encoded.

## C. GitHub Repository

**GitHub Repository Link:** [(https://github.com/bhagyashree-n-m/online-shoppers-classification)]

The repository contains the Jupyter Notebook used for data exploration, preprocessing, model training, and evaluation. It also includes the required dependencies, saved model files, evaluation results, and project documentation.

## D. Models Used and Evaluation Metrics

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models were evaluated using the same train/test split and preprocessing workflow.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.849959448499594 | 0.896199917597049 | 0.510714285714285 | 0.748691099476439 | 0.607218683651804 |0.533040568214008 |
| Decision Tree | 0.85117599351176 | 0.701359648682055 | 0.52112676056338 | 0.484293193717277 | 0.502035278154681 | 0.415073193159251 |
| kNN | 0.873073803730738 | 0.781760935976927 | 0.645569620253164 | 0.400523560209424 | 0.494345718901453 | 0.442196075215932 |
| Naive Bayes | 0.673560421735604 | 0.793161057571524 | 0.293658536585365 | 0.787958115183246 | 0.427860696517412 | 0.323430608166789 |
| Support Vector Machine | 0.856042173560421 | 0.885570439448905 | 0.525812619502868 | 0.719895287958115 | 0.607734806629834 | 0.531851887950821 |

### Evaluation Metrics

- **Accuracy:** Measures the proportion of correctly classified shopping sessions.
- **AUC:** Measures how effectively the model distinguishes between purchase and no-purchase sessions across classification thresholds.
- **Precision:** Measures the proportion of sessions predicted as purchases that actually resulted in purchases.
- **Recall:** Measures the proportion of actual purchases correctly identified by the model.
- **F1-score:** Represents the harmonic mean of precision and recall.
- **MCC:** Measures the agreement between actual and predicted classifications, accounting for true positives, true negatives, false positives, and false negatives.

Since purchase and non-purchase sessions may be imbalanced, accuracy alone is insufficient for selecting the best-performing model. F1-score, AUC, MCC, precision, and recall should also be considered.

## E. Observations on Model Performance

| ML Model Name | Observation About Model Performance |
|---|---|
| **Logistic Regression** | Achieved an accuracy of 85.00% and the highest AUC (0.8962) among all five models. Recorded a recall of 0.7487, showing good performance in identifying customers who are likely to make a purchase.  |
| **Decision Tree** | Achieved reasonable accuracy but had the lowest AUC (0.7014) and was less effective at identifying the positive class., indicating weaker classification ability. |
| **kNN** | Recorded the highest accuracy (87.31%) and precision (0.6456), but had relatively low recall.|
| **Naive Bayes** | Achieved the highest recall (0.7880) but the lowest accuracy and precision. Its relatively low precision indicates that it also incorrectly classified many non-purchasing customers as potential purchasers.|
| **Support Vector Machine** | Achieved an accuracy of 85.60% and an AUC of 0.8856, indicating good overall classification performance and a strong ability to distinguish between the two classes.Its recall of 0.7199 indicates that it successfully identifies a considerable proportion of actual purchasing customers.SVM provides a good balance between precision and recall. |
| **Overall Winner for the Dataset** | SVM performs slightly better in accuracy, precision and F1 Score, while Logistic Regression performs better in AUC, recall and MCC making both the models good performer for the dataset |

### Use the test_data.csv for uploading and testing of the application.
