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
from PIL import Image, ImageOps

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
def create_image_label_pair(IN_IMG,MASK):
    image_file_list = os.listdir(IN_IMG)
    label_file_list = os.listdir(MASK)
    # print(f"number of image files {len(image_file_list)}")
    # print(f"number of label files {len(label_file_list)}")
    assert len(image_file_list)>0,"no files present"
    assert len(label_file_list)>0,"no files present"
    assert all(img[-4:] == ".jpg" for img in image_file_list),"non jpeg files"
    assert all(img[-4:] == ".jpg" for img in label_file_list),"non jpeg files"
    image_file_pair_list = [(a,b) for a in image_file_list for b in label_file_list if a == b]
        
    return image_file_pair_list



def border_image(file_pairs):
    for image,mask in file_pairs:    
        mask_name = mask
        image_name = image
        image = io.imread(os.path.join(IN_IMG,image_name))
        mask = io.imread(os.path.join(MASK,mask_name))
        
            
        # increase the border of mask by 1 pixel. This is to provide a clear border for masks on the edge of the image
        if image.shape[0] == 512 and image.shape[1] == 512:
            # open image as PIL object
            image_n = Image.open(os.path.join(IN_IMG,image_name))
            # increase the border of image by 1 pixel
            image_n = ImageOps.expand(image_n, border=1, fill=0)
            # convert PIL image and mask to numpy array
            image_n = np.array(image_n)
        elif image.shape[0] == 514 and image.shape[1] == 514:
            image_n = image    
        else:
            print(f"image shape is {image.shape[0]},{image.shape[1]}")    

        if mask.shape[0] == 512 and mask.shape[1] == 512:
            mask_n = Image.open(os.path.join(MASK,mask_name))
            mask_n = ImageOps.expand(mask_n, border=1, fill=0)
            mask_n = np.array(mask_n)
        elif mask.shape[0] == 514 and mask.shape[1] ==514:
            mask_n = mask
        else:
            print(f"mask shape is {mask_n.shape[0]},{mask_n.shape[1]}")
        labeled_image,count = connected_components(mask_n)
        # print(f"mask unique {np.unique(mask)} for name {mask_name}")           
    
        if os.path.exists(BDR_OUT):
            assert labeled_image.shape[0] == 514, "label shape is not 514 x 514"
            cv2.imwrite(os.path.join(BDR_OUT,mask_name),labeled_image,[int(cv2.IMWRITE_JPEG_QUALITY), 300])
        else:
            print("label image directory not found")
        if os.path.exists(BDR_IMG):
            assert image_n.shape[0] == 514,"image shape is not 514 x 514"
            image_n = cv2.cvtColor(image_n, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(BDR_IMG,mask_name),image_n,[int(cv2.IMWRITE_JPEG_QUALITY), 300])
        else:
            print("bordered image directory not found")    
        print(count)

if __name__ == "__main__":
    sample = "tumor"
    IN_IMG = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/set1_reclustered_images_new/"+sample+"_red/"
    MASK = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/set1_reclustered_images_new/"+sample+"_mask/"
    BDR_OUT = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/set1_reclustered_images_new/"+sample+"_bordered_labels/"
    BDR_IMG = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/set1_reclustered_images_new/"+sample+"_bordered_red/"

    file_pairs = create_image_label_pair(IN_IMG,MASK)
    print(f"number of pairs {len(file_pairs)}")
    border_image(file_pairs)
