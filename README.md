Predicting Nationality of Individuals Using Their Names with Machine Learning
Project Overview
This project aims to predict the nationality of individuals based solely on their names using various machine learning models. The dataset comprises first names and last names collected from Wikipedia and includes names from 18 different nationalities. The models used include Multinomial Naive Bayes, Logistic Regression, Decision Tree Classifier, XGBoost, and K-Nearest Neighbors.

Dataset
The dataset contains 140,316 names from the following nationalities:

Russia
China
Korea
Poland
Scotland
Italy
US/UK
France
Japan
Greece
Spanish
India
Turkey
Indonesia
Vietnam
Czech Republic
Arabic
Dutch
Each nationality contains around 7500 names, ensuring a close to uniform distribution.

Project Structure
.
├── README.md
├── data
│   └── new.csv                # Dataset containing names and their corresponding nationalities
├── models
│   └── nationality_vectorizer.pkl  # Saved CountVectorizer
├── notebooks
│   └── nationality_prediction.ipynb  # Jupyter notebook with all the code
└── requirements.txt           # List of dependencies

Requirements
Python 3.9
pandas
numpy
scikit-learn
xgboost
matplotlib
joblib
eli5

Install the dependencies using:
pip install -r requirements.txt
Data Preparation
Load the dataset:

python
import pandas as pd
df = pd.read_csv('data/new.csv')
Split the data into features and labels:

python

Xfeatures = df['Names']
ylabels = df['Nationality']
Vectorize the features using CountVectorizer:

python

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
Save the vectorizer for future use:

python

import joblib
joblib.dump(cv, open("models/nationality_vectorizer.pkl", "wb"))
Split the dataset into training and testing sets:

python

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)
Model Training and Evaluation
1. Multinomial Naive Bayes
python

from sklearn.naive_bayes import MultinomialNB

nv = MultinomialNB()
nv.fit(x_train, y_train)
accuracy = nv.score(x_test, y_test)
print(f"Multinomial Naive Bayes Accuracy: {accuracy}")
2. Logistic Regression
python

from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(max_iter=1000)
logit.fit(x_train, y_train)
accuracy = logit.score(x_test, y_test)
print(f"Logistic Regression Accuracy: {accuracy}")
3. Decision Tree Classifier
python

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)
accuracy = dt_model.score(x_test, y_test)
print(f"Decision Tree Accuracy: {accuracy}")
4. XGBoost
python

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(ylabels)
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

xg_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=18, colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10)
xg_clf.fit(x_train1, y_train1)
accuracy = xg_clf.score(x_test1, y_test1)
print(f"XGBoost Accuracy: {accuracy}")
5. K-Nearest Neighbors
python

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
accuracy = knn_model.score(x_test, y_test)
print(f"K-Nearest Neighbors Accuracy: {accuracy}")
Model Evaluation
The models are evaluated using accuracy and classification reports.

python

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

models = [("Multinomial NB", nv), ("Logistic Regression", logit), ("Decision Tree", dt_model), ("XGBoost", xg_clf), ("KNN", knn_model)]

results = []
for model_name, model in models:
    accuracy, report = evaluate_model(model, x_test, y_test)
    results.append({"Model": model_name, "Accuracy": accuracy, "Classification Report": report})

results_df = pd.DataFrame(results)
print(results_df)
Feature Importance
Using eli5 to display the feature importance for the Logistic Regression model.

python

import eli5
eli5.show_weights(logit, target_names=class_names)
Sample Predictions
Making predictions using the trained models:

python

sample_names = ["krishna", "vipond rodden", "chinnadurai itty", "constance andrews", "Le Van Ky Du"]
vect_samples = cv.transform(sample_names).toarray()

print("Predictions using Multinomial Naive Bayes:", nv.predict(vect_samples))
print("Predictions using Logistic Regression:", logit.predict(vect_samples))
print("Predictions using Decision Tree:", dt_model.predict(vect_samples))
print("Predictions using XGBoost:", label_encoder.inverse_transform(xg_clf.predict(vect_samples)))
print("Predictions using K-Nearest Neighbors:", knn_model.predict(vect_samples))
Conclusion
This project demonstrates how various machine learning models can be trained to predict the nationality of individuals based on their names with varying degrees of accuracy. Logistic Regression performed the best, followed by Multinomial Naive Bayes and Decision Tree.
