#importing the libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

#read the csv dataset
data = pd.read_csv('https://raw.githubusercontent.com/SaneerCheenu/Heart-Disease-Prediction/main/heart.csv')

# Rename columns
column_mapping = {
    "sex ": "sex",
    "fbs ": "fbs",
    "restecg ": "restecg",
    "slope ": "slope",
    "chol ": "chol",
    "age ": "age",
    "trestbps ": "trestbps",
    "exang ": "exang",
    "ca ": "ca",
    "oldpeak ": "oldpeak",
    "thal ": "thal"
}

data = data.rename(columns=column_mapping)

data.columns
data.head()


## Feature selection
#get correlation of each feature in dataset
plt.figure(figsize=(17,6))

#plot heat map
g=sns.heatmap(data.corr(),annot=True,cmap="RdYlGn")
# data=data.drop(['fbs', 'chol', 'trestbps', 'restecg'], axis=1)  # axis = 1 means column, axis=0 means rows
target=data['target']
data = data.drop(['target'],axis=1)
data.head()


# We split the data into training and testing set:
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2,random_state=42)


# Random Forest Classifier
classifierRF= RandomForestClassifier(random_state=1)
classifierRF.fit(x_train, y_train)
classifierRF.score(x_test, y_test)

#Saving the model to disk
filename = 'random_forest_model.pkl'
current_directory = os.path.dirname(__file__)
full_path = os.path.join(current_directory, filename)
pickle.dump(classifierRF, open(full_path, 'wb'))

# get predictions from model
y_preds = classifierRF.predict(x_test)
print('Random Forest accuracy score: ',accuracy_score(y_test, y_preds))

cmx = confusion_matrix(y_test,y_preds)
print(cmx)
print('\n')

sns.heatmap(cmx/np.sum(cmx), annot=True, fmt='.2%', cmap='Blues')

print('\n')
print(classification_report(y_test, y_preds))