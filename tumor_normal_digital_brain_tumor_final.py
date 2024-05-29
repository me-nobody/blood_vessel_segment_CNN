# discriminate tumor vs normal
import numpy as np
import pandas as pd

import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy import unique
from numpy import where

from sklearn.ensemble import RandomForestClassifier

from scipy import stats
import joblib

from sklearn.metrics import precision_recall_fscore_support

logger.info("libraries loaded")


normal = pd.read_csv("/users/ad394h/Documents/tumor_cell_classify/normal_objects.csv")
tumor = pd.read_csv("/users/ad394h/Documents/tumor_cell_classify/tumor_objects.csv")

normal = normal.sample(n=19997)
tumor = tumor.sample(n=19997)

tumor[["type"]] = "tumor"
normal[["type"]] = "normal"

# merge the 2 samples
combined_nuclei = pd.concat([tumor,normal],axis=0)

# reset the index
combined_nuclei = combined_nuclei.reset_index(drop=True)

# creating composite indices
combined_nuclei = combined_nuclei[['Nucleus: Area','Nucleus: Circularity', 'Nucleus: Hematoxylin OD mean','type']]

# reset the index
combined_nuclei = combined_nuclei.reset_index(drop=True)

logger.info(f"combined nuclei size {combined_nuclei.shape[0]}")
#logger.info("combined nuclei columns ",combined_nuclei.columns)

# Identify input and target columns
input_cols, target_col = combined_nuclei.columns[:-1], combined_nuclei.columns[-1]
input_df, input_targets = combined_nuclei[input_cols].copy(), combined_nuclei[target_col].copy()

# we will one-hot encode the target column
input_targets_num = input_targets.map({'tumor':1,'normal':0})

# Create training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    input_df, input_targets_num, test_size=0.25, random_state=42)


# save the index and column information
xtrain_idx = X_train.index
xtrain_col = X_train.columns
xtest_idx = X_test.index

ytrain_idx = y_train.index
ytest_idx = y_test.index

# Impute and scale numeric columns
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

X_train.index = xtrain_idx
X_train.columns = xtrain_col

X_test.index = xtest_idx
X_test.columns = xtrain_col

logger.info(f"train dataset dimensions {X_train.shape[0]}, {X_train.shape[1]}")
logger.info(f"test dataset dimensions {X_test.shape[0]}, {X_test.shape[1]}")
#remove outliers

# Calculate the z-score for each feature
z_area = np.abs(stats.zscore(X_train['Nucleus: Area']))
z_circle =  np.abs(stats.zscore(X_train['Nucleus: Circularity']))
z_dye = np.abs(stats.zscore(X_train['Nucleus: Hematoxylin OD mean']))

X_train['z_area'] = z_area
X_train['z_circle'] = z_circle
X_train['z_dye'] = z_dye

threshold =3

X_train = X_train[X_train['z_area']< threshold]
X_train = X_train[X_train['z_circle']< threshold]
X_train = X_train[X_train['z_dye']< threshold]

# filter y_train values corresponding to X_train values
y_train = y_train[X_train.index]

X_train.shape, y_train.shape

if all(X_test.index == y_test.index):
    logger.info(f"X_test and y test indices are same")

X_train = X_train[['Nucleus: Area', 'Nucleus: Circularity', 'Nucleus: Hematoxylin OD mean']]

#now that we have obtained the best possible parameters{'max_depth': 5, 'max_features': 1, 'min_samples_leaf': 20, 'n_estimators': 100}, we shall re-run the classifier.

clf = RandomForestClassifier(max_depth=5, random_state=0,min_samples_leaf=20,n_estimators=100)

clf.fit(X_train,y_train)


#save the RandomForest Classifier

import joblib

# save classifier to onedrive
# joblib.dump(clf, "/users/ad394h/Documents/models/tumor_RandomForest_classifier.joblib")

predicted_probabilities = pd.DataFrame(clf.predict_proba(X_test))
predicted_probabilities.set_index(ytest_idx,inplace=True)

#logger.info("predicted_probabilities",predicted_probabilities.value_counts())


logger.info(f"test tumor samples taken {y_test.value_counts()[0]}")
logger.info(f"test normal samples taken {y_test.value_counts()[1]}")


# collect all predictions with >0.75 probability 
predicted_probabilities_50 = predicted_probabilities[(predicted_probabilities[1]>0.51)]
predicted_probabilities_75 = predicted_probabilities[(predicted_probabilities[1]>0.60)]
# match the indices of y_test and predictions
y_test75 = y_test[predicted_probabilities_75.index]

x_train_cols = X_train.columns.tolist()

logger.info(f"Based on  {x_train_cols[0]}, {x_train_cols[1]}, {x_train_cols[2]}")

logger.info(f"total test tumor values predicted as tumor with >51% precision {predicted_probabilities_50.shape[0]}")

logger.info(f"total test tumor values predicted as tumor with >60% precision {predicted_probabilities_75.shape[0]}")

logger.info(f"y_test correctly classified as tumor {y_test75.value_counts()[1]}")
logger.info(f"y_test misclassified as tumor  {y_test75.value_counts()[0]}")

tumor_percent = (y_test75.value_counts()[1]/y_test.value_counts()[1])*100

normal_percent = (y_test75.value_counts()[0]/y_test.value_counts()[1])*100

logger.info(f"predicted tumor percent {tumor_percent}")
logger.info(f"predicted misclassified as normal percent {normal_percent}")

# create a matrix of ones same as y_test75
y_pred75 = np.ones_like(y_test75)

if y_test75.shape==y_pred75.shape:
    logger.info("test and prediction indices match")

precision_stats = precision_recall_fscore_support(y_test75, y_pred75, average='weighted')
logger.info(f"precision {precision_stats[0]} recall {precision_stats[1]} F-score {precision_stats[2]}")

