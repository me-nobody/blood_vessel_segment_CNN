
import numpy as np
import pandas as pd

import cv2

import time
import os

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# access google drive
IMAGE_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/images'
logger.info(f'number of image files {os.listdir(IMAGE_PATH).sort()}')

MASK_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/masks'
logger.info(f'number of mask files {os.listdir(MASK_PATH).sort()}')


# we have to resize the images to a specific shape

class Images():

  """resize the images to the desired shape"""

  def __init__(self, img_path, mask_path):
      self.img_path = img_path
      self.mask_path = mask_path
      

  def resize(self, size= None):
    img_x = size # divisible by 32 for the CNN to work
    img_y = size # divisible by 32 for hte CNN to work
    if size is not None:
      for file in os.listdir(self.img_path).sort(): # sorting the list is IMPORTANT
          img = os.path.join(img_path, file)
          logger.info(f' image is {img}')

      for file in os.listdir(self.mask_path).sort():
          msk = os.path.join(mask_path,file)
          logger.info(f'mask is {file}')

    else:
        return 0

    # img = cv2.imread(self.img_path + self.X[idx] + '.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,(img_x,img_y),cv2.INTER_LINEAR)

    # #img = Image.fromarray(img)
    # mask = cv2.imread(self.mask_path + self.X[idx] + '.png')
    # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    # mask = cv2.resize(mask,(img_x,img_y),cv2.INTER_LINEAR)
    # # we are re-mapping the mask
    
    # mask = np.tile(mask,(3,1,1)) # same size as input image
    # mask = np.moveaxis(mask,0,2) # same shape as input image

     


# transformed_img_list,transformed_msk_list = get_augmented_images(slide_dataset)

# for i,pair in enumerate(zip(transformed_img_list,transformed_msk_list)):
#   image_name = "img_"+str(i+1)+".png"
#   mask_name = "img_"+str(i+1)+".png"
#   image_path = os.path.join(AUG_IMG_PATH,image_name)
#   mask_path = os.path.join(AUG_MSK_PATH,image_name)
#   cv2.imwrite(image_path,pair[0])
#   cv2.imwrite(mask_path,pair[1])


