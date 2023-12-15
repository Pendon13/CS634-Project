from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from sklearn.metrics import RocCurveDisplay
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from pathlib import Path
Path("./model/binary").mkdir(parents=True, exist_ok=True)

name = 'eec'
# name = 'eec' - EasyEnsembleClassifier
# name = 'randomforest' - BalancedRandomForestClassifier


# We change the following to binary
# No radiation and/or cancer-directed surgery is 0
# Some form of surgery is 1

# y_enc_classes = ["No radiation and/or cancer-directed surgery",
#                 "Intraoperative rad with other rad before/after surgery",
#                 "Intraoperative radiation",
#                 "Radiation after surgery",
#                 "Radiation before and after surgery",
#                 "Radiation prior to surgery",
#                 "Sequence unknown, but both were given",
#                 "Surgery both before and after radiation"
#                 ]

y_enc_classes = ["No radiation and/or cancer-directed surgery",
                 "Surgery given"]

print("loading...")

X = pd.read_csv("2020-extra_removed_preprocessed.csv", header=0, dtype=int)
y = pd.read_csv("2020-extra_removed_preprocessed_target_binary.csv", header=0, dtype=int)
y = y[["0"]].values.ravel()


# Assuming 'X' is your feature matrix and 'y' is your target variable

# Split your data into training and testing sets
print("split")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print("model start")

# Create and train the EasyEnsembleClassifier
if name == 'eec':
    clf = EasyEnsembleClassifier(random_state=0, n_estimators=10, n_jobs=2)

# Create and train the BalancedRandomForestClassifier
if name == 'randomforest':
    clf = BalancedRandomForestClassifier(
        n_estimators=100, random_state=0, sampling_strategy="all", replacement=True
    )


try:
    print('try loading')
    clf = joblib.load('./model/binary/mlp-' + str(name) + '-main.joblib')
    timeTaken = 0
    print('load success')
except:
    print('fitting')
    start_time = time.time()
    clf.fit(X_train, y_train)
    timeTaken = time.time()-start_time
    print(timeTaken)
    print('fit done')
    try:
        print('saving clf')
        joblib.dump(clf, './model/binary/mlp-' + str(name) + '-main.joblib')
    except:
        print('failed to joblib dump')

y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
clf_params = clf.get_params()
clf_acc = balanced_accuracy_score(y_test, y_pred)

f = open("./model/binary/clf-y_pred-" + str(name) + ".txt",'a')
for i in y_pred:
    f.write(str(i) + '\n')
f.close()
f = open("./model/binary/clf-y_test-" + str(name) + ".txt",'a')
for i in y_test:
    f.write(str(i) + '\n')
f.close()

f = open("./model/binary/clf-" + str(name) + ".txt",'a')
f.write(str(clf_params)+"\n")
f.write("predict_probab" + str(y_score) +"\n")
f.write(str(balanced_accuracy_score(y_test, y_pred)))
f.write("time: " + str(timeTaken))
f.write("#"*50+"\n")
f.close()

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(cm).plot()
cm_display.figure_.savefig('./model/binary/clf-cm-'+ str(name) + '-main.png')

from sklearn.metrics import RocCurveDisplay, roc_curve

fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
roc_display.figure_.savefig('./model/binary/clf-roc-'+ str(name) + '-main.png')

from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
pr_display.figure_.savefig('./model/binary/clf-precrec-'+ str(name) + '-main.png')