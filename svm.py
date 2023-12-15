from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn import svm

y_enc_classes = ["Intraoperative rad with other rad before/after surgery",
                "Intraoperative radiation",
                "No radiation and/or cancer-directed surgery",
                "Radiation after surgery",
                "Radiation before and after surgery",
                "Radiation prior to surgery",
                "Sequence unknown, but both were given",
                "Surgery both before and after radiation"
                ]

print("loading...")
X = pd.read_csv("2020-extra_removed_preprocessed.csv", header=0, dtype=int)
y = pd.read_csv("2020-extra_removed_preprocessed_target.csv", header=0, dtype=int)
y = y[["0"]].values.ravel()
print("split")
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("model start")
from sklearn.svm import SVC

print('making model')
clf = svm.SVC(decision_function_shape='ovr',
              gamma = 1,
              kernel = 'sigmoid',
              class_weight = 'balanced', # inverse proportional balancing
              max_iter = 100000,

              )
print("fit to model")

clf.fit(X, y)

clf_params = clf.get_params()
# X_pred = clf.predict(X_test)
X_score = clf.score(X_train, y_train)
y_score = clf.predict_proba(X_test)
# dec_one = clf.decision_function([[1]])
f = open("clf-svm.txt",'a')
f.write(str(clf_params)+"\n")
f.write('The number of iterations was:',clf.n_iter_)
f.write("\npredict_proba" + str(y_score))
f.write("\nscore: "+ str(X_score)+"\n")
f.write("#"*50+"\n")
f.close()

import pickle
from joblib import dump, load
dump(clf, './model/svm-1.joblib')