import numpy as np
import pandas as pd
import os
#
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import sklearn as sk

os.chdir('C:/Users/beomj/PycharmProjects/Lineage/data')


def classification_model(model, data, predictors, outcome):
    # Fit the model :
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    #Perform K-fold cross-validation with 5 folds
    kf = KFold(n_splits=5)
    error =[]
    for train, test in kf.split(data):
        # Filter training data
        train_predictors = (data[predictors.iloc[train,:]])

        # The target we're using to train the algorithm
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

        print("cross-validation score : %s" % "{0:.3%}".format(np.mean(error)))

        # Fit the model again so that it can be refered outside the function:
        model.fit(data[predictors], data[outcome])



print(sk.__version__)

df = pd.read_csv('myWork0.csv')
label = pd.read_csv('train_label.csv')
df['survival_time'] = label['survival_time']
df = df.drop('acc_id', axis =1)
traindf, testdf = train_test_split(df, test_size=0.3)

print(df.head())


predictor_var = list(df.columns)
predictor_var.remove('survival_time')
outcome_var = 'survival_time'
model = DecisionTreeClassifier()


classification_model(model, traindf, predictor_var, outcome_var)
