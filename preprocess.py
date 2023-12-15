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

# LabelEncoder gets classes and puts them in alphabetical order
y_enc_classes = ["Intraoperative rad with other rad before/after surgery",
                "Intraoperative radiation",
                "No radiation and/or cancer-directed surgery",
                "Radiation after surgery",
                "Radiation before and after surgery",
                "Radiation prior to surgery",
                "Sequence unknown, but both were given",
                "Surgery both before and after radiation"
                ]

# prepare input data
def one_hot_enc_input(X_input):
    oe = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
    oe.fit(X_input)
    X_train_enc = oe.transform(X_input)
    #X_input_values = oe.get_feature_names_out()
    return X_train_enc#, X_input_values
 
# prepare target
def prepare_target(y_train):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    #classes_single = list(le.inverse_transform([0,1,2,3,4,5,6,7]))
    return y_train_enc#, classes_single

def load_dataset(filename):
    data = pd.read_csv(filename, header=0,dtype=object)

    ySet = data[["RX Summ--Surg/Rad Seq"]]
    y = prepare_target(ySet.values.ravel())
    y = pd.DataFrame(y)
    print(y.head())
    # Change to Binary Classification
    # Label "No radiation and/or cancer-directed surgery" as 0
    # Label everything else as 1

    y[0].replace(0, 1, inplace = True)
    y[0].replace(3, 1, inplace = True)
    y[0].replace(4, 1, inplace = True)
    y[0].replace(5, 1, inplace = True)
    y[0].replace(6, 1, inplace = True)
    y[0].replace(7, 1, inplace = True)
    y[0].replace(2, 0, inplace = True)

    data = data.drop(columns=["RX Summ--Surg/Rad Seq", "Primary Site", "Histologic Type ICD-O-3"], axis=1)
    print(data.columns)
    # Preprocess Age
    data["Age recode with <1 year olds"].replace("00 years", 0, inplace=True)
    data["Age recode with <1 year olds"].replace("01-04 years", 1, inplace=True)
    for i in range(8):
        year_one = str(i)+"0-"+str(i)+"4 years"
        year_five = str(i)+"5-"+str(i)+"9 years"
        value_one = i*10
        value_five = i*10 + 5
        data["Age recode with <1 year olds"].replace(year_one, value_one, inplace=True)
        data["Age recode with <1 year olds"].replace(year_five, value_five, inplace=True)

    data["Age recode with <1 year olds"].replace("80-84 years", 80, inplace=True)
    data["Age recode with <1 year olds"].replace("85+ years", 85, inplace=True)

    # Preprocess Sex
    data["Sex"].replace(['Female','Male'], [0,1], inplace=True)

    # Preprocess Months from Diagnosis to treatment
    data["Months from diagnosis to treatment"].replace("Blank(s)", -1, inplace=True)

    #race_cols, race_classes = one_hot_enc_input(data[["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"]])
    race_cols = one_hot_enc_input(data[["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"]])
    race_df = pd.DataFrame(race_cols)
    data = pd.concat([data, race_df], axis=1).drop(columns=["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"], axis=1)

    # tnm_cols, tnm_classes = one_hot_enc_input(data[["TNM 7/CS v0204+ Schema recode"]])
    tnm_cols = one_hot_enc_input(data[["TNM 7/CS v0204+ Schema recode"]])
    tnm_df = pd.DataFrame(tnm_cols)
    data = pd.concat([data, tnm_df], axis=1).drop(columns=["TNM 7/CS v0204+ Schema recode"], axis=1)

    # site_cols, site_classes = one_hot_enc_input(data[["Site recode ICD-O-3 2023 Revision"]])
    site_cols = one_hot_enc_input(data[["Site recode ICD-O-3 2023 Revision"]])
    site_df = pd.DataFrame(site_cols)
    data = pd.concat([data, site_df], axis=1).drop(columns=["Site recode ICD-O-3 2023 Revision"], axis=1)

    # site_exp_cols, site_exp_classes = one_hot_enc_input(data[["Site recode ICD-O-3 2023 Revision Expanded"]])
    site_exp_cols = one_hot_enc_input(data[["Site recode ICD-O-3 2023 Revision Expanded"]])
    site_exp_df = pd.DataFrame(site_exp_cols)
    data = pd.concat([data, site_exp_df], axis=1).drop(columns=["Site recode ICD-O-3 2023 Revision Expanded"], axis=1)

    # cs_schema_cols, cs_schema_classes = one_hot_enc_input(data[["CS Schema - AJCC 6th Edition"]])
    cs_schema_cols = one_hot_enc_input(data[["CS Schema - AJCC 6th Edition"]])
    cs_schema_df = pd.DataFrame(cs_schema_cols)
    data = pd.concat([data, cs_schema_df], axis=1).drop(columns=["CS Schema - AJCC 6th Edition"], axis=1)

    # primary_cols, primary_classes = one_hot_enc_input(data[["Primary Site - labeled"]])
    primary_cols = one_hot_enc_input(data[["Primary Site - labeled"]])
    primary_df = pd.DataFrame(primary_cols)
    data = pd.concat([data, primary_df], axis=1).drop(columns=["Primary Site - labeled"], axis=1)

    # icd_cols, icd_classes = one_hot_enc_input(data[["ICD-O-3 Hist/behav"]])
    icd_cols = one_hot_enc_input(data[["ICD-O-3 Hist/behav"]])
    icd_df = pd.DataFrame(icd_cols)
    data = pd.concat([data, icd_df], axis=1).drop(columns=["ICD-O-3 Hist/behav"], axis=1)

    # icd_mali_cols, icd_mali_classes = one_hot_enc_input(data[["ICD-O-3 Hist/behav, malignant"]])
    icd_mali_cols = one_hot_enc_input(data[["ICD-O-3 Hist/behav, malignant"]])
    icd_mali_df = pd.DataFrame(icd_mali_cols)
    data = pd.concat([data, icd_mali_df], axis=1).drop(columns=["ICD-O-3 Hist/behav, malignant"], axis=1)

    # site_sirs_cols, site_sirs_classes = one_hot_enc_input(data[["Site recode ICD-O-3/WHO 2008 (for SIRs)"]])
    site_sirs_cols = one_hot_enc_input(data[["Site recode ICD-O-3/WHO 2008 (for SIRs)"]])
    site_sirs_df = pd.DataFrame(site_sirs_cols)
    data = pd.concat([data, site_sirs_df], axis=1).drop(columns=["Site recode ICD-O-3/WHO 2008 (for SIRs)"], axis=1)

    # eod_cols, eod_classes = one_hot_enc_input(data[["EOD Schema ID Recode (2010+)"]])
    eod_cols = one_hot_enc_input(data[["EOD Schema ID Recode (2010+)"]])
    eod_df = pd.DataFrame(eod_cols)
    data = pd.concat([data, eod_df], axis=1).drop(columns=["EOD Schema ID Recode (2010+)"], axis=1)

    # tumor_rare_cols, tumor_rare_classes = one_hot_enc_input(data[["Site recode - rare tumors"]])
    tumor_rare_cols = one_hot_enc_input(data[["Site recode - rare tumors"]])
    tumor_rare_df = pd.DataFrame(tumor_rare_cols)
    data = pd.concat([data, tumor_rare_df], axis=1).drop(columns=["Site recode - rare tumors"], axis=1)

    # tumor_size_cols, tumor_size_classes = one_hot_enc_input(data[["Tumor Size Summary (2016+)"]])
    tumor_size_cols = one_hot_enc_input(data[["Tumor Size Summary (2016+)"]])
    tumor_size_df = pd.DataFrame(tumor_size_cols)
    data = pd.concat([data, tumor_size_df], axis=1).drop(columns=["Tumor Size Summary (2016+)"], axis=1)

    # region_nodes_cols, region_nodes_classes = one_hot_enc_input(data[["Regional nodes positive (1988+)"]])
    region_nodes_cols = one_hot_enc_input(data[["Regional nodes positive (1988+)"]])
    region_nodes_df = pd.DataFrame(region_nodes_cols)
    data = pd.concat([data, region_nodes_df], axis=1).drop(columns=["Regional nodes positive (1988+)"], axis=1)

    # site_malin_cols, site_malin_classes = one_hot_enc_input(data[["Site - mal+ins (most detail)"]])
    site_malin_cols = one_hot_enc_input(data[["Site - mal+ins (most detail)"]])
    site_malin_df = pd.DataFrame(site_malin_cols)
    data = pd.concat([data, site_malin_df], axis=1).drop(columns=["Site - mal+ins (most detail)"], axis=1)

    # site_mal_cols, site_mal_classes = one_hot_enc_input(data[["Site - malignant (most detail)"]])
    site_mal_cols = one_hot_enc_input(data[["Site - malignant (most detail)"]])
    site_mal_df = pd.DataFrame(site_mal_cols)
    data = pd.concat([data, site_mal_df], axis=1).drop(columns=["Site - malignant (most detail)"], axis=1)

    X = data
    
    return X, y

print("loading...")#

X, y = load_dataset("2020-extra_removed.csv")
X.to_csv("2020-extra_removed_preprocessed.csv")
y.to_csv('2020-extra_removed_preprocessed_target_binary.csv')
