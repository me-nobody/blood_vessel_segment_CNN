import numpy as np
import tifffile as tif
import os
import skimage as ski
from skimage import io
from skimage import feature
import pandas as pd
import cv2
import histomicstk as htk
import skimage.io
import skimage.measure
import skimage.color
from skimage.color import rgb2hed, hed2rgb
from skimage.morphology import remove_small_objects


IN_IMG = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/margin_images/ub19_52388_2_set1_512px/images/"
LBL_IMG = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/margin_images/ub19_52388_2_set1_512px/BGR2RGB_vessel_mask/"
OUT = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/margin_images/ub19_52388_2_set1_512px/dump/"

# create label image
def connected_components(mask, sigma=1.0, t=200, connectivity=1):
    # denoise the image with a Gaussian filter
    # blurred_mask = ski.filters.gaussian(mask, sigma=sigma)    
    # print(f"blurred_mask {np.unique(blurred_mask)}")
    # mask the image according to threshold
    # binary_mask = blurred_mask > t        
    binary_mask = mask > t
    # perform connected component analysis
    labeled_mask, count = ski.measure.label(binary_mask, connectivity=connectivity, return_num=True)
    # return labeled_image, count
    # cv2.imwrite(os.path.join(OUT,"labeled_mask.jpg"),labeled_mask*10,[int(cv2.IMWRITE_JPEG_QUALITY), 300])
    return labeled_mask,count

# labeled_image,num_objects = connected_components(subtracted_mask, sigma=1.0, t=0.5, connectivity=2)

# create pairs for sending to histomics image feature generator
def create_image_label_pair(IN_IMG,LBL_IMG):
    image_file_list = os.listdir(IN_IMG)
    label_file_list = os.listdir(LBL_IMG)
    # print(f"number of image files {len(image_file_list)}")
    # print(f"number of label files {len(label_file_list)}")
    assert len(image_file_list)>0,"no files present"
    assert len(label_file_list)>0,"no files present"
    assert all(img[-4:] == ".jpg" for img in image_file_list),"non jpeg files"
    assert all(img[-4:] == ".jpg" for img in label_file_list),"non jpeg files"
    image_file_pair_list = [(a,b) for a in image_file_list for b in label_file_list if a == b]
        
    return image_file_pair_list

file_pairs = create_image_label_pair(IN_IMG,LBL_IMG)
# print(f"number of pairs {len(file_pairs)}")

for image,mask in file_pairs:    
    mask_name = mask
    image = io.imread(os.path.join(IN_IMG,image))
    mask = io.imread(os.path.join(LBL_IMG,mask))      
    # print(f"mask unique {np.unique(mask)} for name {mask_name}")   
    labeled_image,count = connected_components(mask)
    cv2.imwrite(os.path.join(OUT,mask_name),labeled_image,[int(cv2.IMWRITE_JPEG_QUALITY), 300])
    print(count)