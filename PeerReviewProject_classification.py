import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

###read data in


file = 'Human_Activity_Recognition_Using_Smartphones_Data.csv'

data = pd.read_csv(file,header = 0)

data.dtypes.value_counts()

data.describe().T

data.Activity.value_counts()

#check if float fields needs scaling: No
data.iloc[:,:-1].min().value_counts()
data.iloc[:,:-1].max().value_counts()

###split data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

features = [x for x in data.columns if x != 'Activity']

le = LabelEncoder()
target = pd.Series(le.fit_transform(data['Activity']))

X_train,X_test,y_train,y_test = train_test_split(data[features],target,test_size = 0.3,random_state = 42)

###logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model1 = LogisticRegression(solver = 'liblinear').fit(X_train,y_train)
fit1 = model1.predict(X_test)

error_list = list()

error_list.append(pd.Series({'LogisticRegression':accuracy_score(y_test,fit1)}))

###decision tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

###default decision tree

dt = DecisionTreeClassifier(random_state= 42)

dt.fit(X_train,y_train)

param0 = {'max_depth': range(1,dt.tree_.max_depth+1,2),
          'max_features': range(1,20)}

GV_DT = GridSearchCV(DecisionTreeClassifier(random_state= 42),
                     param_grid = param0,
                     scoring= 'accuracy',
                     n_jobs = -1)

GV_DT.fit(X_train,y_train)

model2 =GV_DT.best_estimator_.fit(X_train,y_train)

fit2 = model2.predict(X_test)

error_list.append(pd.Series({'DecisionTree':accuracy_score(y_test,fit2)}))

###random forest
from sklearn.ensemble import RandomForestClassifier

n_trees = [100,150,200,250]
max_features = [1,2,3,4]



GV_RF = GridSearchCV(RandomForestClassifier(oob_score = True,
                            random_state = 42,
                            warm_start = True,
                            n_jobs = -1),
                  param_grid = {'n_estimators': n_trees},
                  scoring = 'accuracy',
                  n_jobs = -1)

GV_RF.fit(X_train,y_train)

model3 = GV_RF.best_estimator_.fit(X_train,y_train)

fit3 = model3.predict(X_test)


error_list.append(pd.Series({'RandomForest':accuracy_score(y_test,fit3)}))

###gradient boost
from sklearn.ensemble import GradientBoostingClassifier


param2 = {'n_estimators': n_trees,
          'max_features': max_features,
          'learning_rate':[0.1,0.01,0.001,0.0001],
          'subsample':[1.0,0.5]}

GV_GBC = GridSearchCV(GradientBoostingClassifier(random_state = 42),
                  param_grid = param2,
                  scoring = 'accuracy',
                  n_jobs = -1)

GV_GBC.fit(X_train,y_train)

model4 = GV_GBC.best_estimator_.fit(X_train,y_train)

fit4 = model4.predict(X_test)


error_list.append(pd.Series({'GradientBoosting':accuracy_score(y_test,fit4)}))

###adaboost
from sklearn.ensemble  import AdaBoostClassifier

ABC  = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1,random_state = 42))


param3 = {'n_estimators': n_trees,
          
          'learning_rate':[0.1,0.001]}

GV_ABC = GridSearchCV(ABC,
                  param_grid = param3,
                  scoring = 'accuracy',
                  n_jobs = -1)

GV_ABC.fit(X_train,y_train)

model5 = GV_ABC.best_estimator_.fit(X_train,y_train)

fit5 = model5.predict(X_test)


error_list.append(pd.Series({'AdaBoosting':accuracy_score(y_test,fit5)}))

error_list
###Accuracy results
#LogisticRegression    0.978964
#DecisionTree    0.907767
#RandomForest    0.976052
#GradientBoosting    0.983172
#AdaBoosting    0.711974

from sklearn.metrics import confusion_matrix, roc_curve

cm1 = confusion_matrix(y_test,fit1)

cm2 = confusion_matrix(y_test,fit2)

cm3 = confusion_matrix(y_test,fit3)

cm4 = confusion_matrix(y_test,fit4)

cm5 = confusion_matrix(y_test,fit5)

ax1 = sns.heatmap(cm1,annot = True,fmt = 'd')
ax1.set_title('Logistic Regression')

ax2 = sns.heatmap(cm2,annot = True,fmt = 'd')
ax2.set_title('DecisionTree')

ax3 = sns.heatmap(cm3,annot = True,fmt = 'd')
ax3.set_title('RandomForest')

ax4 = sns.heatmap(cm4,annot = True,fmt = 'd')
ax4.set_title('GradientBoosting')

ax5 = sns.heatmap(cm5,annot = True,fmt = 'd')
ax5.set_title('Adaboosting')
