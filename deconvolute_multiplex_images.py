import numpy as np
import pandas as pd
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
from skimage import morphology
from skimage.color import rgb2hed, hed2rgb
from skimage.morphology import remove_small_objects
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
    estimate_sigma,
)

IN_IMG = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/margin_images/ub19_52388_2_set1_512px/images/"
LBL_IMG = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/margin_images/ub19_52388_2_set1_512px/BGR2RGB_vessel_mask/"
OUT1 = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/margin_images/ub19_52388_2_set1_512px/set1_red/"
OUT2 = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/margin_images/ub19_52388_2_set1_512px/set1_dab/"
OUT3 = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/margin_images/ub19_52388_2_set1_512px/set1_green/"
OUT4 = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/margin_images/ub19_52388_2_set1_512px/"

# create stain color map
stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
print('stain_color_map:', stain_color_map, sep='\n')


# one set of stain colors deciphered from QuPath
stain_color_dict = {"Red":[0.733, 0.51, 0.45],
					"Green":[0.413,0.665, 0.622],
					"DAB":[0.268, 0.570, 0.776],
                    "Null":[0.0,0.0,0.0],
					"White":[1.0,1.0,1.0]}

# assign the reference color dictionary to htk
htk.preprocessing.color_deconvolution.stain_color_map = stain_color_dict
# assign htk dictionary to an object
stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
print('custom stain_color_map:', stain_color_map, sep='\n')

# specify stains of input image
stain_1 = ['Green',
          'DAB',
          'Red']

# create the stain matrix
W = np.array([stain_color_map[colr] for colr in stain_1]).T




# create pairs for sending to histomics image feature generator
def create_image_label_pair(IN_IMG,LBL_IMG):
    assert os.path.exists(IN_IMG),"image path does not exist"
    assert os.path.exists(LBL_IMG),"label path does not exist"
    image_file_list = os.listdir(IN_IMG)
    label_file_list = os.listdir(LBL_IMG)
    print(f"number of image files {len(image_file_list)}")
    print(f"number of label files {len(label_file_list)}")
    assert len(image_file_list)>0,"no files present"
    assert len(label_file_list)>0,"no files present"
    assert all(img[-4:] == ".jpg" for img in image_file_list),"non jpeg files"
    assert all(img[-4:] == ".jpg" for img in label_file_list),"non jpeg files"
    image_file_pair_list = [(a,b) for a in image_file_list for b in label_file_list if a == b]
        
    return image_file_pair_list


def deconvolute():
    file_pairs = create_image_label_pair(IN_IMG,LBL_IMG)
    print(f"number of pairs {len(file_pairs)}")
    r_sigma_list=[]
    dab_sigma_list=[]
    g_sigma_list=[]
    for image,mask in file_pairs:    
        mask_name = mask
        image_name = image
        image = io.imread(os.path.join(IN_IMG,image))
        mask = io.imread(os.path.join(LBL_IMG,mask))      
        # deconvolute the stains
        imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(image, W) # map matrix
        # first layer
        multiplex_r = imDeconvolved.Stains[:, :, 0]
        # multiplex_r = ski.filters.gaussian(multiplex_r, sigma=0.05) # gaussian filtering
        multiplex_r = ski.util.invert(multiplex_r)
        r_sigma = estimate_sigma(multiplex_r)
        r_sigma_list.append(r_sigma)
        # multiplex_r = htk.preprocessing.color_conversion.od_to_rgb(multiplex_r) # convert OD to RGB
        # multiplex_r = (multiplex_r * 255).astype(np.uint16) # convert to 16-bit
        # print(f"r shape {multiplex_r.shape}")
        r_name = image_name[:-4]+"_rr.jpg"
        cv2.imwrite(os.path.join(OUT1,r_name),multiplex_r,[int(cv2.IMWRITE_JPEG_QUALITY), 300])

        # second layer
        multiplex_dab = imDeconvolved.Stains[:, :, 1]
        # multiplex_dab = ski.filters.gaussian(multiplex_dab, sigma=0.05)
        multiplex_dab = ski.util.invert(multiplex_dab)
        dab_sigma = estimate_sigma(multiplex_dab)
        dab_sigma_list.append(dab_sigma)
        # specific denoising
        # multiplex_dab =  denoise_bilateral(multiplex_dab, sigma_color=0.05, sigma_spatial=15, channel_axis=-1,multichannel=False)
        # multiplex_dab = htk.preprocessing.color_conversion.od_to_rgb(multiplex_dab) # convert OD to RGB
        # multiplex_dab = (multiplex_dab * 255).astype(np.uint16) # convert to 16-bit
        dab_name = image_name[:-4]+"_dd.jpg"
        cv2.imwrite(os.path.join(OUT2,dab_name),multiplex_dab,[int(cv2.IMWRITE_JPEG_QUALITY), 300])

        # third_layer
        multiplex_g = imDeconvolved.Stains[:, :, 2]
        # multiplex_g = ski.filters.gaussian(multiplex_g, sigma=0.05)
        multiplex_g = ski.util.invert(multiplex_g)
        g_sigma = estimate_sigma(multiplex_g)
        g_sigma_list.append(g_sigma)
        # multiplex_g = htk.preprocessing.color_conversion.od_to_rgb(multiplex_g)
        # multiplex_g = (multiplex_g * 255).astype(np.uint16) # convert to 16-bit
        g_name = image_name[:-4]+"_gg.jpg"      
        cv2.imwrite(os.path.join(OUT3,g_name),multiplex_g,[int(cv2.IMWRITE_JPEG_QUALITY), 300])
    sigma_df = pd.DataFrame({"r_sigma":r_sigma_list,"dab_sigma":dab_sigma_list,"g_sigma":g_sigma_list})    
    sigma_df.to_csv(os.path.join(OUT4,"sigma.csv"))
    
deconvolute()    