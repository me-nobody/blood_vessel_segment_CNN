# discriminate tumor vs normal
# code run on MARS
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

logger.info("libraries loaded")

import joblib

detection_objects = pd.read_csv("/users/ad394h/Documents/tumor_cell_classify/objects.txt", sep="\t")

detection_objects.set_index('Object ID',inplace=True)

logger.info("indices set")
detection_objects = detection_objects[['Nucleus: Area', 'Nucleus: Circularity', 'Nucleus: Hematoxylin OD mean']]

sample_cols = detection_objects.columns

sample_index = detection_objects.index

detection_objects = pd.DataFrame(scaler.fit_transform(detection_objects))

detection_objects.set_index(sample_index,inplace=True)

detection_objects.columns = sample_cols

logger.info(f"the number of detection objects are {detection_objects.shape[0]} with {detection_objects.shape[1]} parameters")

#load the classifier
clf = joblib.load('/users/ad394h/Documents/models/tumor_RandomForest_classifier.joblib')
logger.info("classifier loaded")

# predict the classes for the 2 samples
detection_object_probabilities = pd.DataFrame(clf.predict_proba(detection_objects))

# match the index as the original samples
detection_object_probabilities.set_index(sample_index,inplace=True)

# select those predictions where we are >75% confident of being tumor cells
# the threshold for tumor cells is being kept lower, to highlight zones of tumor better
predicted_tumor_60 = detection_object_probabilities[(detection_object_probabilities[1]>0.60)]
predicted_normal_75 = detection_object_probabilities[(detection_object_probabilities[0]>0.75)]

logger.info(f"the number of the predicted tumor cells is {predicted_tumor_60.shape[0]}")
logger.info(f"the number of the predicted stroma cells are {predicted_normal_75.shape[0]}")

predicted_tumor_60.columns = ['normal','tumor']
predicted_normal_75.columns = ['normal','tumor']

predicted_tumor_ids = predicted_tumor_60.index.tolist()
predicted_normal_ids = predicted_normal_75.index.tolist()

logger.info(f"length of tumor indexes {len(predicted_tumor_ids)}")
logger.info(f"length of stroma indexes {len(predicted_normal_ids)}")

tumor_percent = (predicted_tumor_60.shape[0]/detection_object_probabilities.shape[0])*100

normal_percent = (predicted_normal_75.shape[0]/detection_object_probabilities.shape[0])*100

logger.info(f"predicted tumor percent {tumor_percent}")
logger.info(f"predicted normal percent {normal_percent}")

predicted_tumor_60.to_csv("/users/ad394h/Documents/tumor_cell_classify/predict75_tumor_object_probs.csv")
predicted_normal_75.to_csv("/users/ad394h/Documents/tumor_cell_classify/predict75_normal_object_probs.csv")

fh1 = open('/users/ad394h/Documents/tumor_cell_classify/predict75_tumor_object_ids.txt','a')
fh1.flush()
for id in predicted_tumor_ids:
  fh1.write(id)
  fh1.write("\n")
fh1.close()

fh2 = open('/users/ad394h/Documents/tumor_cell_classify/predict75_normal_object_ids.txt','a')
fh2.flush()
for id in predicted_normal_ids:
  fh2.write(id)
  fh2.write("\n")
fh2.close()


