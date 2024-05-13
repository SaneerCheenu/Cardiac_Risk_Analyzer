#importing the libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, roc_auc_score
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
classifierRF= RandomForestClassifier(max_depth=5, random_state=1)
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

def eval_model(model, x_train, x_test, y_train, y_test):
    """
    Function to evaluate the given model based on Train and test data.

    """
    eval_df = pd.DataFrame(index=['RFClassifier'])

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    eval_df['Train Accuracy'] = accuracy_score(y_train, y_train_pred)
    eval_df['Test Accuracy'] = accuracy_score(y_test, y_test_pred)

    eval_df['Train ROC AUC Score'] = roc_auc_score(y_train, y_train_pred)
    eval_df['Test ROC AUC Score'] = roc_auc_score(y_test, y_test_pred)

    eval_df['Train F1 Score'] = f1_score(y_train, y_train_pred)
    eval_df['Test F1 Score'] = f1_score(y_test, y_test_pred)

    eval_df['Train Precision Score'] = precision_score(y_train, y_train_pred)
    eval_df['Test Precision Score'] = precision_score(y_test, y_test_pred)

    eval_df['Train Recall Score'] = recall_score(y_train, y_train_pred)
    eval_df['Test Recall Score'] = recall_score(y_test, y_test_pred)

    return eval_df.T

eval_df = eval_model(classifierRF, x_train, x_test, y_train, y_test)
eval_df

plt.figure(figsize=(15, 8))
colors = sns.color_palette('pastel')
ax = sns.barplot(data=eval_df, x=eval_df.index, y='RFClassifier', palette=colors)
for container in ax.containers:
    ax.bar_label(container)
plt.title('Evaluation Metrics Plot')
plt.tight_layout()
plt.show()