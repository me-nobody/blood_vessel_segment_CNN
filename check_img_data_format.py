import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


import time
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/images/'
MASK_PATH = '/users/ad394h/Documents/segment_blood_vessels/data/masks/'
TEST_PATH = '/users/ad394h/Documents/segment_blood_vessels/tests/'

def create_df(PATH):
    name = []
    img_shapes = []
    for dirname, _, filenames in os.walk(PATH):
        for filename in filenames:
            name.append(filename[:-4])
            img = cv2.imread(os.path.join(IMAGE_PATH,filename))
            if img is not None:
                img_shape = img.shape
                img_shapes.append(img_shape)
            else:
                logger.info(f'{filename} is not an image')

    return pd.DataFrame({'id': name,'shape':img_shapes}, index = np.arange(0, len(name)))

image_df = create_df(IMAGE_PATH)
#image_df.to_csv(TEST_PATH+"image_shape.csv")
logger.info(f'Total Images: {len(image_df)}')

mask_df = create_df(MASK_PATH)
#mask_df.to_csv(TEST_PATH+"mask_shape.csv")
logger.info(f'Total Images: {len(mask_df)}')