from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import re


# read titanic train and test data
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

# copied from: https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset
full_data = [train, test]

# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# Remove all NULLS in the Age column
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)
X = train.drop(['Survived'], axis=1)
y = train["Survived"]
X_test = test.drop(['Survived'], axis=1)
y_test = test["Survived"]
def accuracy(y_gt, y_pred):
    return np.sum(y_gt == y_pred) / y_gt.shape[0]


from decision_tree import DecisionTree

#dt = DecisionTree(criterion='gini', max_depth=6, min_samples_leaf=2, sample_feature=False)

# TODO: Train the best DecisionTree(best val accuracy) that you can. You should choose some 
# hyper-parameters such as critertion, max_depth, and min_samples_in_leaf 
# according to the cross-validation result.
# To reduce difficulty, you can use KFold here.
#from sklearn.model_selection import KFold
#kf = KFold(n_splits=5, shuffle=True, random_state=2020)
#criters=['infogain_ratio','gini']
#depths=[2, 3, 4]
#leaves=[1, 4, 8, 16, 32, 64]
#estimators=[10, 20, 50]
#best_criter=criters[0]
#best_depth=depths[0]
#best_leaf=leaves[0]
#best_estimator=estimators[0]
#best_score=0.0


#from random_forest import RandomForest
#for depth in depths:
#    for leaf in leaves:
#        for criter in criters:
#            for estimator in estimators:
#                print('{} {} {} {}'.format(depth, leaf, criter, estimator))
#                base_learner = DecisionTree(criterion=criter, max_depth=depth, min_samples_leaf=leaf, sample_feature=False)
#                rf = RandomForest(base_learner=base_learner, n_estimator=estimator, seed=2020)
#                pred=np.zeros(len(X))
#                for train_indice, valid_indice in kf.split(X, y):
#                    X_train_fold, y_train_fold = X.loc[train_indice], y.loc[train_indice]
#                    X_val_fold, y_val_fold = X.loc[valid_indice], y.loc[valid_indice]
#                    # print(X_train_fold.shape, X_val_fold.shape)
#                    rf.fit(X_train_fold, y_train_fold)
#                    pred[valid_indice]=rf.predict(X_val_fold)
#                score=accuracy(y, pred)
#                if score>best_score:
#                    best_score=score
#                    best_criter=criter
#                    best_depth=depth
#                    best_leaf=leaf
#                    best_estimator=estimator
#                print("Accuracy on validation set: {}".format(score))

#print('{} {} {} {}'.format(best_criter, best_depth, best_leaf, best_estimator))
#base_learner = DecisionTree(criterion=best_criter, max_depth=best_depth, min_samples_leaf=best_leaf, sample_feature=True)
#rf = RandomForest(base_learner=base_learner, n_estimator=best_estimator, seed=2020)
## end answer
#rf.fit(X, y)
#print("Accuracy on train set: {}".format(accuracy(y, rf.predict(X))))
#print("Accuracy on test set: {}".format(accuracy(y_test, rf.predict(X_test))))

from GBDT import GBDT
from regression_tree import RegressionTree

#base_learner = RegressionTree(max_depth=4, min_samples_leaf=4, sample_feature=False)
#gbdt = GBDT(base_learner=base_learner, learning_rate=0.12, n_estimator=120)
#gbdt.fit(X, y)

#y_train_pred = gbdt.predict(X)

#print("Accuracy on train set: {}".format(accuracy(y, y_train_pred)))

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=2020)
depths=[3, 4, 5, 6]
estimators=[10, 20, 50, 100, 200]
learning_rates=[0.1,0.2,0.5]
best_depth=depths[0]
best_estimator=estimators[0]
best_lr=learning_rates[0]
best_score=0.0
for depth in depths:
    for estimator in estimators:
        for lr in learning_rates:
            print('{} {} {}'.format(depth, estimator, lr))
            base_learner = RegressionTree(max_depth=depth, min_samples_leaf=pow(2, depth-1), sample_feature=False)
            gbdt = GBDT(base_learner=base_learner, learning_rate=lr, n_estimator=estimator)
            pred=np.zeros(len(X))
            for train_indice, valid_indice in kf.split(X, y):
                X_train_fold, y_train_fold = X.loc[train_indice], y.loc[train_indice]
                X_val_fold, y_val_fold = X.loc[valid_indice], y.loc[valid_indice]
                # print(X_train_fold.shape, X_val_fold.shape)
                gbdt.fit(X_train_fold, y_train_fold)
                pred[valid_indice]=gbdt.predict(X_val_fold)
            score=accuracy(y, pred)
            if score>best_score:
                best_score=score
                best_lr=lr
                best_depth=depth
                best_estimator=estimator
            print("Accuracy on validation set: {}".format(score))
print('{} {} {} {}'.format(best_depth, best_estimator, best_lr))
base_learner = RegressionTree(max_depth=best_depth, min_samples_leaf=pow(2, best_depth-1), sample_feature=False)
gbdt = GBDT(base_learner=base_learner, learning_rate=best_lr, n_estimator=best_estimator)
# end answer
gbdt.fit(X, y)
print("Accuracy on train set: {}".format(accuracy(y, gbdt.predict(X))))
print("Accuracy on test set: {}".format(accuracy(y_test, gbdt.predict(X_test))))

#from random_forest import RandomForest

#base_learner = DecisionTree(criterion='entropy', max_depth=3, min_samples_leaf=1, sample_feature=True)
#rf = RandomForest(base_learner=base_learner, n_estimator=100, seed=2020)
#rf.fit(X, y)

#y_train_pred = rf.predict(X)

#print("Accuracy on train set: {}".format(accuracy(y, y_train_pred)))
#print("Accuracy on test set: {}".format(accuracy(y_test, rf.predict(X_test))))

#from adaboost import Adaboost

##learners=[]
##for criter in criters:
##    for depth in depths:
##        learners.append(DecisionTree(criterion=criter, max_depth=depth, min_samples_leaf=pow(2, depth-1), sample_feature=False))

#base_learner=DecisionTree(criterion='infogain_ratio', max_depth=4, min_samples_leaf=32, sample_feature=False)

#ada = Adaboost(base_learner=base_learner, n_estimator=10, seed=2020)
#ada.fit(X, y)

#y_train_pred = ada.predict(X)

#print("Accuracy on train set: {}".format(accuracy(y, y_train_pred)))

#for estimator in estimators:
#    ada = Adaboost(base_learners=learners, n_estimator=estimator, seed=2020)
#    pred=np.zeros(len(X))
#    for train_indice, valid_indice in kf.split(X, y):
#        X_train_fold, y_train_fold = X.loc[train_indice], y.loc[train_indice]
#        X_val_fold, y_val_fold = X.loc[valid_indice], y.loc[valid_indice]
#        # print(X_train_fold.shape, X_val_fold.shape)
#        ada.fit(X_train_fold, y_train_fold)
#        pred[valid_indice]=ada.predict(X_val_fold)
#    score=accuracy(y, pred)
#    if score>best_score:
#        best_score=score
#        best_estimator=estimator
#    print("Accuracy on validation set: {}".format(score))

#print(best_estimator)
#ada = Adaboost(base_learners=learners, n_estimator=best_estimator, seed=2020)
#ada.fit(X, y)
#print("Accuracy on train set: {}".format(accuracy(y, ada.predict(X))))
#print("Accuracy on test set: {}".format(accuracy(y_test, ada.predict(X_test))))

#from sklearn.model_selection import cross_val_score
#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import GradientBoostingClassifier

#models=[GaussianNB(),
#        Perceptron(),
#        LogisticRegression(C=0.06),
#        SVC(C=4,gamma=0.015),
#        MLPClassifier(alpha=1),
#        KNeighborsClassifier(n_neighbors=8),
#        DecisionTreeClassifier(),
#        RandomForestClassifier(n_estimators=500),
#        AdaBoostClassifier(n_estimators=500), 
#        GradientBoostingClassifier(n_estimators=120,learning_rate=0.12,max_depth=4)]
#names=['NB','Perception','LR','SVM','NN','KNN','DS','RF','Ada','GBDT']
#for name,model in zip(names,models):
#    score=cross_val_score(model,X,y,cv=5)
#    print("{}: {},{}".format(name,score.mean(),score))