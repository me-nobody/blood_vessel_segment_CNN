import os, sys
import cv2
import shutil

import numpy as np
import pandas as pd
import PIL
from PIL import Image

import time
import logging

timestr = time.strftime("%Y%m%d-%H%M%S")

log_file = '/users/ad394h/Documents/multiplex/logs/sm_unet_effi_c2d3l64r512_predict{}.txt'.format(timestr)


logging.basicConfig(filename=log_file,level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# get the the logger with the name 'PIL'
pil_logger = logging.getLogger('PIL')  
# override the logger logging level to INFO
pil_logger.setLevel(logging.INFO)


# 2 class Unet model

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    logger.info("No GPU available. Training will run on CPU.")

unet_model = torch.load('/users/ad394h/Documents/multiplex/model/unet_c2d3l64r512.pt')    

PATH = '/users/ad394h/Documents/multiplex/data/core_multiplex_512px'

# create subdirectories for storing masks
def create_directories(PATH):    
    if os.path.exists(PATH):
        for root,subdir,files in os.walk(PATH):        
            if subdir:
                if any([dir.endswith("mask") for dir in subdir]):
                    break            
                else:
                    for dir in subdir:
                        new_dir = dir+"_mask"
                        new_path = os.path.join(PATH,new_dir)                
                        os.mkdir(new_path)
                        os.chdir(new_path)
                        os.mkdir(os.path.join(new_path,"image"))
                        os.mkdir(os.path.join(new_path,"mask"))

# predict mask of actual images
def predict_image_mask(model, image):
    img_x = 512 # same size as in trained model
    img_y = 512
    
    image = cv2.resize(image,(img_x,img_y),cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # comment out for masks of alternate channels 
    model.eval()
    image = torch.from_numpy(image).float()    
    image = normalize(image,dim=0)
    image = torch.permute(image,(2,0,1))

    model.to(device); image=image.to(device)

    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)        
        masked = torch.argmax(output, dim=1)        
        masked = masked.cpu().squeeze(0)
        masked = masked.numpy()
        masked = np.where(masked==1,255,0)
    return output,masked                        

create_directories(PATH)
os.chdir(PATH)      
if os.path.exists(PATH):
    for root,subdir,files in os.walk(PATH):                
        if subdir:
            mask_dir = [dir  for dir in subdir if dir.endswith("mask") and dir.startswith("ub")]
            parent_dir = [dir for dir in subdir if not dir.endswith("mask") and dir.startswith("ub")]
            parent_dir.sort()
            mask_dir.sort()
            for parent,mask in zip(parent_dir,mask_dir):
                parent_path = os.path.join(PATH,parent)
                mask_path = os.path.join(PATH,mask)
                parent_file_list = os.listdir(parent_path)
                for file in parent_file_list:
                    file_path = os.path.join(parent_path,file)
                    image = cv2.imread(file_path)
                    output,masked = predict_image_mask(model = unet_model,image = image)
                    image = cv2.resize(image,(512,512))
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    file = file[:-4] + ".jpg"
                    cv2.imwrite(os.path.join(mask_path,"image",file),image)
                    cv2.imwrite(os.path.join(mask_path,"mask",file),masked)