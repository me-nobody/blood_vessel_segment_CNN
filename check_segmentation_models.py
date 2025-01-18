import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import os
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/segment_blood_vessels/logs/check_segmentation_models_{}.txt'.format(timestr)


logging.basicConfig(filename=log_file, level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




import segmentation_models_pytorch as smp

# model = smp.Unet(
#     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,
#     activation = None,               # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     encoder_depth =5,
#     decoder_channels = [256,128,64,32,16],
#     classes=3,                      # model output channels (number of classes in your dataset)
# )



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import normalize

# import cv2

from sklearn.model_selection import train_test_split


model = torch.load('/users/ad394h/Documents/segment_blood_vessels/models/models_2_test/Unet_efficientnet_b7_noweights.pt')


for child in model.named_children():
    logger.info(child)








