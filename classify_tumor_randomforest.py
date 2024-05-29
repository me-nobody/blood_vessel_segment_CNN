# classify tumor cells
import numpy as np
import pandas as pd

import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from numpy import unique
from numpy import where

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import joblib

logger.info("libraries loaded")

# tumor_sample = pd.read_csv("/users/ad394h/Documents/tumor_cell_classify/gbm_2_objects.txt", sep="\t")
ihc_sample = pd.read_csv("/users/ad394h/Documents/tumor_cell_classify/7433_2022_10_27_objects.txt", sep="\t")

# we set the object ID as index to correlate with QuPath
#tumor_sample.set_index('Object ID',inplace=True)
ihc_sample.set_index('Object ID',inplace=True)

logger.info("indices set")

# tumor_sample.head(n=3)
#tumor_sample = tumor_sample[['Nucleus: Area', 'Nucleus: Circularity', 'Nucleus: Hematoxylin OD mean']]
ihc_sample = ihc_sample[['Nucleus: Area', 'Nucleus: Circularity', 'Nucleus: Hematoxylin OD mean']]

# save row indexes and columns 
sample_cols = ihc_sample.columns
#tumor_sample_index = tumor_sample.index
ihc_sample_index = ihc_sample.index

# scale the dataset
#tumor_sample = pd.DataFrame(scaler.fit_transform(tumor_sample))
ihc_sample = pd.DataFrame(scaler.fit_transform(ihc_sample))

# set the index the same as the input file, to correlate with the original image
#tumor_sample.set_index(tumor_sample_index,inplace=True)
ihc_sample.set_index(ihc_sample_index,inplace=True)

#tumor_sample.columns = sample_cols
ihc_sample.columns = sample_cols

# load the classifier
clf = joblib.load('/users/ad394h/Documents/models/tumor_RandomForest_classifier.joblib')
logger.info("classifier loaded")

# predict the classes for the 2 samples
#tumor_predicted_probabilities = pd.DataFrame(clf.predict_proba(tumor_sample))
ihc_predicted_probabilities = pd.DataFrame(clf.predict_proba(ihc_sample))

# match the index as the original samples
#tumor_predicted_probabilities.set_index(tumor_sample_index,inplace=True)
ihc_predicted_probabilities.set_index(ihc_sample_index,inplace=True)

# select those predictions where we are >75% confident of being tumor cells
#tumor_predicted_75 = tumor_predicted_probabilities[(tumor_predicted_probabilities[1]>0.75)]
ihc_predicted_75 = ihc_predicted_probabilities[ihc_predicted_probabilities[1]>0.75]

#tumor_predicted_75.shape, ihc_predicted_75.shape, tumor_predicted_probabilities.shape, ihc_predicted_probabilities.shape
#tumor_predicted_75.head(n=3)

#percent_tumor_DBT = (tumor_predicted_75.shape[0]/tumor_sample.shape[0])*100

#logger.info(f"of {tumor_sample.shape[0]} tumor cell objects detected from GBM sample of Digital Brain Tumor database {percent_tumor_DBT} % classified as tumor")

percent_tumor_IHC = (ihc_predicted_75.shape[0]/ihc_sample.shape[0])*100

logger.info(f"of {ihc_sample.shape[0]} tumor cell objects detected from IHC 7434_2022_10_27 slide {percent_tumor_IHC} % classified as tumor")

#tumor_predicted_75.columns = ['normal','tumor']
ihc_predicted_75.columns = ['normal','tumor']

# save the predictions as csv
ihc_predicted_75.to_csv("/users/ad394h/Documents/tumor_cell_classify/ihc_7433_2022_object_predictions.csv")
#tumor_predicted_75.to_csv("/users/ad394h/Documents/tumor_cell_classify/digital_brain_atlas_tumor_object_predictions.csv")

logger.info("csv file created")

ihc_object_ids = ihc_predicted_75['Object ID'].tolist()

fh = open('/users/ad394h/Documents/tumor_cell_classify/ihc_7433_2022_object_ids.txt','a')
for id in ihc_object_ids:
  fh.write(id)
  fh.write("\n")
fh.close()
