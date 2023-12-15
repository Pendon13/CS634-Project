from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from sklearn.metrics import RocCurveDisplay
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from pathlib import Path
Path("./model/multi/").mkdir(parents=True, exist_ok=True)
#hidden_layer = (8, 3, 8)
#hidden_layer = (1963,)
hidden_layer = (841, 241, 69, 20, 6, 2)

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


# Assuming 'X' is your feature matrix and 'y' is your target variable

# Split your data into training and testing sets
print("split")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print("model start")
# Create and train the MLPClassifier
n_iter_max = 100000
start_time = time.time()
clf = MLPClassifier(hidden_layer_sizes=hidden_layer, 
                    max_iter=n_iter_max, 
                    alpha = 0.0001, 
                    activation='relu', 
                    solver='adam', 
                    random_state=42, 
                    tol=1e-15)
try:
    print('try loading')
    clf = joblib.load('./model/multi/mlp-'+ str(hidden_layer) + '-main.joblib')
    timeTaken = 0
    print('load success')
except:
    print('fitting')
    with joblib.parallel_backend('threading', n_jobs=2):
        clf.fit(X_train, y_train)
    timeTaken = time.time()-start_time
    print(timeTaken)
    print('fit done')
    try:
        print('saving clf')
        joblib.dump(clf, './model/multi/mlp-'+ str(hidden_layer) + '-main.joblib')
    except:
        print('failed to joblib dump')


y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
clf_params = clf.get_params()
clf_score = clf.score(X_test, y_test)


f = open("./model/multi/clf-mlp-y_pred" + str(hidden_layer) + ".txt",'a')
for i in y_pred:
    f.write(str(i) + '\n')
f.close()
f = open("./model/multi/clf-mlp-y_test" + str(hidden_layer) + ".txt",'a')
for i in y_test:
    f.write(str(i) + '\n')
f.close()

f = open("./model/multi/clf-mlp-" + str(hidden_layer) + ".txt",'a')
f.write(str(clf_params)+"\n")
f.write("predict_probab" + str(y_pred_proba) +"\n")
f.write(classification_report(y_test, y_pred))
f.write('\nThe number of iterations was: ' + str(clf.n_iter_))  # the number of iterations
f.write('\nThe final loss is ' + str(clf.loss_))            # total loss
f.write('\nThe output activation is ' + str(clf.out_activation_))
f.write("\nscore: "+ str(clf_score)+"\n")
f.write("time: " + str(timeTaken))
f.write("#"*50+"\n")
f.close()

# Initialize an empty dictionary to store fpr, tpr, and roc_auc for each class
roc_auc_dict = {}

n_classes = len(y_enc_classes)
target_names = y_enc_classes
y_onehot_test = y_test
y_score = clf_score

from sklearn.metrics import auc, roc_curve
from itertools import cycle

fig, ax = plt.subplots(figsize=(6, 6))

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for class_id, color in zip(range(n_classes), colors):
    RocCurveDisplay.from_predictions(
        y_onehot_test[:],
        y_score[:],
        name=f"ROC curve for {target_names[class_id]}",
        color=color,
        ax=ax,
        plot_chance_level=(class_id == 2),
    )

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
plt.legend()
plt.show()
