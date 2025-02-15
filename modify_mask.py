import numpy as np
import os,sys,time
from tqdm import tqdm
import cv2

PATH = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/core_images/"

if os.path.exists(PATH):
    for root,subfolders,files in os.walk(PATH):
       if subfolders:
           subfolders = subfolders
           break

def send_files(subfolders):    
    for folder in subfolders:
        filepath = os.path.join(PATH,folder,"BGR2RGB")
        # print(filepath)
        files = os.listdir(filepath)
        for file in files:
            yield filepath,file
   
OUTPATH = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/output/"   
def update_mask(file):
    mask = cv2.imread(file)
    mask = mask[:,:,0]
    mask = np.where(mask<10,0,255)
    mask = mask.astype(np.uint8)
    # print(np.unique(mask),mask.shape)   
    # Define a 3x3 square structuring element
    # input mask has to be unit8 dtype
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Dilate the mask by 2 pixels
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    # Erode the mask by 5 pixels (applying erosion 5 times)
    eroded_mask = cv2.erode(mask, kernel, iterations=5)

    # create outline mask by substracting the dilated_mask from the eroded mask
    subtracted_mask = cv2.subtract(dilated_mask, eroded_mask)
    return mask,subtracted_mask
catch  = send_files(subfolders)    
for file_tuple in catch:
    # print(file_tuple[0])
    file = os.path.join(file_tuple[0],file_tuple[1])
    outfile_path_1 = os.path.join(file_tuple[0][:-8],"BGR2RGB_vessel_mask",file_tuple[1])
    outfile_path_2 = os.path.join(file_tuple[0][:-8],"BGR2RGB_full_mask",file_tuple[1])
    # print(outfile_path_1)
    # print(outfile_path_2)
    mask,subtracted_mask = update_mask(file)
    cv2.imwrite(outfile_path_1,subtracted_mask)
    cv2.imwrite(outfile_path_2,mask)