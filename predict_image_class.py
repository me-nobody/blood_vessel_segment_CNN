import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.nn.functional import normalize
import cv2
from glob import glob
import os
from skimage import io
import time

import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/microvascular_proliferation/logs/predict_image_class_NN_{}.txt'.format(timestr)


logging.basicConfig(filename=log_file,level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# plt.set_loglevel (level = 'warning')

# get the the logger with the name 'PIL'
logger = logging.getLogger('PIL')  
# override the logger logging level to INFO
logger.setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    logger.info("No GPU available. Training will run on CPU.")

# access image path


model = torch.load('/users/ad394h/Documents/microvascular_proliferation/model/nn_classification_mvp_trained_180125.pt')


# function to read images from directory and create a df
def create_prediction_df(IMAGE_PATH):  
  image_list = []
  for root, _, images in os.walk(IMAGE_PATH):
    images_ = images
    path = root    
  img_name = [name[:-4] for name in images_]  
  path = [path for i in range(len(images_))]  
  sample_df = pd.DataFrame({'image': img_name,'path':path},index = np.arange(0, len(img_name)))
  return sample_df

# predict one image
def predict_image_mask(model, image,idx=None):
    img_x = 224 # same size as in trained model
    img_y = 224
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(img_x,img_y),cv2.INTER_LINEAR)
    dict_classes = {0: 'background', 1: 'tumor'}
    model.eval()
    image = torch.from_numpy(image).float()
    image = normalize(image,dim=0)
    image = torch.permute(image,(2,0,1))
    img_id = idx
    model.to(device); image=image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        _, preds = torch.max(output, 1)
        preds = preds.cpu().detach().numpy()[0]
        preds_text = dict_classes[preds]
        # print(f"image {img_id} is predicted {preds}")
        return preds,preds_text
    
# function to predict mvp from images in a dataframe
def predict_mvp(model, df):
  image_list = []
  preds_list = []
  preds_text_list = []  
  for index, row in df.iterrows():
    image = io.imread(row['path']+row['image']+".jpg")
    try:
      preds,preds_text = predict_image_mask(model, image,row['image'])
      logger.info(f"{index} image processed")  
    except Exception as e:
      logger.info(f"{e} while predicting")    
    image_list.append(row['image'])
    preds_list.append(preds)
    preds_text_list.append(preds_text)
  pred_df = pd.DataFrame({'image': image_list,'prediction':preds_list,'preds_text':preds_text_list},index = np.arange(0, len(image_list)))
  return pred_df    

if __name__ == "__main__":
  IMAGE_PATH = '/users/ad394h/Documents/microvascular_proliferation/data/ub19_49455_2_split_images/'
  logger.info(f' the image file numbers are {len(os.listdir(IMAGE_PATH))}')

  OUTPUT_PATH = '/users/ad394h/Documents/microvascular_proliferation/logs/'

  filename = "predicted_mvp_class_ub19_49455_2_" + str(timestr) + ".csv"

  try:
      sample_df = create_prediction_df(IMAGE_PATH)      
      predicted_df = predict_mvp(model,sample_df)
      predicted_df.to_csv(os.path.join(OUTPUT_PATH,filename),index=False)
  except Exception as e:
      logger.info(f"{e} in charting history")
