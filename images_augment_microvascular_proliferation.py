import numpy as np
import PIL
from PIL import Image
import cv2
import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import time
import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/microvascular_proliferation/logs/mvp_albumentations_{}.txt'.format(timestr)


logging.basicConfig(filename=log_file,level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# get the the logger with the name 'PIL'
pil_logger = logging.getLogger('PIL')  
# override the logger logging level to INFO
pil_logger.setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    logger.info("No GPU available. Training will run on CPU.")


# access google drive
POS_IMAGE_PATH = '/users/ad394h/Documents/microvascular_proliferation/data/mvp/'
logger.info(f' the mvp file numbers are {len(os.listdir(POS_IMAGE_PATH))}')

NEG_IMAGE_PATH = '/users/ad394h/Documents/microvascular_proliferation/data/other/'
logger.info(f'the other file numbers are {len(os.listdir(NEG_IMAGE_PATH))}')


AUG_POS_IMAGE_PATH = '/users/ad394h/Documents/microvascular_proliferation/data/augmented_mvp/'

AUG_NEG_IMAGE_PATH = '/users/ad394h/Documents/microvascular_proliferation/data/augmented_other/'


# create augmented image folders
if not os.path.exists(AUG_POS_IMAGE_PATH): 
  os.makedirs(AUG_POS_IMAGE_PATH) 

if not os.path.exists(AUG_NEG_IMAGE_PATH): 
  os.makedirs(AUG_NEG_IMAGE_PATH) 

OUTPUT_PATH = '/users/ad394h/Documents/microvascular_proliferation/logs/'

# create a numpy array of image names

def create_array(image_path):  
  for dirname, _, img_names in os.walk(image_path):
    img_names = img_names
  img_names.sort()

  image_array = np.array(img_names)  
  return image_array

mvp_array = create_array(POS_IMAGE_PATH)

other_array = create_array(NEG_IMAGE_PATH)



"""### DATASET"""

class SlideDataset(Dataset):
    def __init__(self, img_path, image_array, transform=None, patch=False):
        self.img_path = img_path        
        self.X = image_array
        self.transform = transform
        self.patches = patch

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
      img_x = 1024
      img_y = 1024
      img = cv2.imread(self.img_path + self.X[idx])
      # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = cv2.resize(img,(img_x,img_y),cv2.INTER_LINEAR)
      return img

# datasets
mvp_set = SlideDataset(POS_IMAGE_PATH,mvp_array)
other_set = SlideDataset(NEG_IMAGE_PATH,other_array)


# library for augmenting images- Albumentations library as A
# Albumentations has pixel level and spatial level transformations
t_train = A.Compose([A.HorizontalFlip(p=0.3),
                     A.VerticalFlip(p=0.3),
                     A.GridDistortion(p=0.1),
                     A.OpticalDistortion(p=0.1),
                     A.Sharpen(p=0.2),
                     A.CLAHE(p=0.1),
                     A.Defocus(p=0.2),
                    #  A.RandomSnow(p=0.2),
                    #  A.Affine(p=0.2,scale=0.8,shear=2.0),
                     A.Affine(p=0.05,scale=1.0,shear=2.0),
                     A.Affine(p=0.05,scale=0.9,shear=1.2),
                     A.SafeRotate(p=0.2),
                     A.ThinPlateSpline(p=0.2),
                    #  A.ShiftScaleRotate(p=0.1),
                     A.Morphological(p=0.3,operation="dilation",scale=[5,5]),
                     A.Morphological(p=0.3,operation="erosion",scale=[5,5])])

def get_augmented_images(slide_dataset):
  img_x = 1024
  img_y = 1024
  len_dataset = len(slide_dataset)
  transformed_img_list =[]
  
  logger.info(f"dataset has {len_dataset} images")
  for idx in range(len_dataset):
    slide_img = slide_dataset.__getitem__(idx)
    slide_img = cv2.resize(slide_img,(img_x,img_y),cv2.INTER_LINEAR)    
    transformed_img_list.append(slide_img)    
    # add the augmentations
    for ndx in range(80):
      transformed = t_train(image = slide_img) # albumentations applied
      t_sl_img = transformed['image']      
      transformed_img_list.append(t_sl_img)      
    logger.info(f"total images in library {len(transformed_img_list)}")    
  return transformed_img_list

mvp_list = get_augmented_images(mvp_set)
other_list = get_augmented_images(other_set)

logger.info(f"mvp in mvp list {mvp_list[0]}")
logger.info(f"other in other list {other_list[0]}")

for i,image in enumerate(mvp_list):
  image_name = "mvp_aug_"+str(i+1)+".jpg"  
  image_path = os.path.join(AUG_POS_IMAGE_PATH,image_name)  
  cv2.imwrite(image_path,image)

offset = len(mvp_list) 

for i,other in enumerate(other_list):
  other_name = "other_aug_"+str(offset+i+1)+".jpg"  
  other_path = os.path.join(AUG_NEG_IMAGE_PATH,other_name)  
  cv2.imwrite(other_path,other)


logger.info(f' the augmented mvp file numbers are {len(os.listdir(AUG_POS_IMAGE_PATH))}')


logger.info(f'the augmented other file numbers are {len(os.listdir(AUG_NEG_IMAGE_PATH))}')