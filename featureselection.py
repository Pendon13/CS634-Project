from sklearn.feature_selection import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from matplotlib import pyplot

def load_dataset(filename):
    data = pd.read_csv(filename, header=0,dtype=str)
    dataset = data.values

    # split into input (X) and output (y) variables
    X = dataset[:, 5:-1]
    y = dataset[:,3]
    print(dataset[1,3])
    X = X.astype(str)
    return X, y

def filewrite(name, line):
    f = open(name,"a")
    f.write(line)
    f.close()

# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc

# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def featureselect(filename):
    print("loading...")
    X, y = load_dataset("./splitdata-targets5/"+filename)
    print("split")
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    print("prepare input")
    # prepare input data
    X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
    print("prepare output")
    # prepare output data
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    print("feature selection")
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)

    # what are scores for the features
    for i in range(len(fs.scores_)):
        line = 'Feature %d: %f' % (i, fs.scores_[i])
        filewrite('./feature-targets5/feature-'+filename, line + "\n")
    
    # plot the scores
    # pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    # pyplot.show()

# featureselect('targets5-0.txt')

for i in range(82):
    featureselect('targets5-'+str(i)+'.txt')