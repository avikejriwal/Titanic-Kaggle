#whether or not to write to output
writeOK = True

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.classifier import StackingClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#PREPROCESS AND FEATURE ENGINEER DATA

train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


titleDict = {"Capt":       "Officer",
            "Col":        "Officer",
            "Major":      "Officer",
            "Jonkheer":   "Royalty",
            "Don":        "Royalty",
            "Sir" :       "Royalty",
            "Dr":         "Officer",
            "Rev":        "Officer",
            "the Countess":"Royalty",
            "Dona":       "Royalty",
            "Mme":        "Mrs",
            "Mlle":       "Miss",
            "Ms":         "Mrs",
            "Mr" :        "Mr",
            "Mrs" :       "Mrs",
            "Miss" :      "Miss",
            "Master" :    "Master",
            "Lady" :      "Royalty"}

train_df['Title'] = train_df['Title'].map(titleDict)
test_df['Title'] = test_df['Title'].map(titleDict)


#Drop useless/missing too many values
train_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)
test_df.drop(['Cabin', 'Name', 'Ticket'], inplace=True, axis=1)

#fill missing values
train_df['Age'] = train_df[['Age', 'Sex', 'Pclass']].groupby(['Sex', 'Pclass']).transform(lambda x:  x.fillna(x.mean() + x.std()*np.random.normal()))
test_df['Age'] = test_df[['Age', 'Sex', 'Pclass']].groupby(['Sex', 'Pclass']).transform(lambda x:  x.fillna(x.mean() + x.std()*np.random.normal()))
#Age depends on sex and passenger class a lot; impute with means based on those values


# train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
# test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())
train_df['Embarked'] = train_df[['Embarked', 'Pclass']].groupby('Pclass').transform(lambda x: x.fillna(x.mode()[0]))
test_df['Fare'] = test_df[['Fare', 'Pclass']].groupby('Pclass').transform(lambda x: x.fillna(x.mean()))


train_df['FamSize'] = train_df['SibSp'] + train_df['Parch']+1
train_df.drop(['SibSp', 'Parch'], inplace=True, axis=1)

test_df['FamSize'] = test_df['SibSp'] + test_df['Parch']+1
test_df.drop(['SibSp', 'Parch'], inplace=True, axis=1)

survived = train_df.pop('Survived')
passId = test_df.pop('PassengerId')

#get one-hot encodings
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

train_cat = train_df.columns.values
test_cat = test_df.columns.values

missingfeats = list(set(train_cat)-set(test_cat))
for feat in missingfeats:
    test_df[feat] = 0

missingfeats = list(set(test_cat)-set(train_cat))
for feat in missingfeats:
    train_df[feat]=0

###YOUR TRAINING BEGINS HERE

xTrain, xVal, yTrain, yVal = train_test_split(train_df, survived, test_size=0.5)

layer1 = LogisticRegression()

models = [ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, criterion='entropy', max_features='log2'),\
        RandomForestClassifier(n_estimators=220, criterion= 'gini', max_depth= 5, max_features='sqrt')]

model = StackingClassifier(classifiers=models, meta_classifier=layer1)

model = RandomForestClassifier(n_estimators=220, criterion= 'gini', max_depth= 5, max_features='sqrt')

print 'Training...'
est = model
est.fit(xTrain, yTrain)
print 'Training accuracy: ', accuracy_score(est.predict(xTrain), yTrain)

print 'Validation accuracy: ', accuracy_score(est.predict(xVal), yVal)

#BUILD AND PREDICT MODEL ON FULL DATASET

if writeOK:
    print 'Predicting...'
    est = model
    est.fit(train_df, survived)
    print 'Train accuracy: ', accuracy_score(est.predict(train_df), survived)

    output = est.predict(test_df)

    predictions_file = open("submission.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(passId, output))
    predictions_file.close()

    print 'Done.'
