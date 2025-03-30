import numpy as np
import pandas as pd
import os
import skimage as ski
from skimage import io
from skimage import feature
import histomicstk as htk
from skimage.morphology import remove_small_objects



def create_sets(GREEN_PATH=None,RED_PATH=None,DAB_PATH=None, LABEL_IMG=None,*label_name):    
    green_file_list = os.listdir(GREEN_PATH)
    red_file_list = os.listdir(RED_PATH)
    dab_file_list = os.listdir(DAB_PATH)
    label_file_list = os.listdir(LABEL_IMG)    
    
    print(f"number of green image files {len(green_file_list)} red image files {len(red_file_list)} dab image files {len(dab_file_list)}")
    print(f"number of label files {len(label_file_list)}")
    
    assert len(green_file_list)>0,"no green files present"
    assert len(red_file_list)>0,"no red files present"
    assert len(dab_file_list)>0,"no dab files present"
    assert len(label_file_list)>0,"no label files present"
    assert all(img[-4:] == ".jpg" for img in green_file_list),"non jpeg files in green"
    assert all(img[-4:] == ".jpg" for img in red_file_list),"non jpeg files in red"
    assert all(img[-4:] == ".jpg" for img in dab_file_list),"non jpeg files in dab"
    assert all(img[-4:] == ".jpg" for img in label_file_list),"non jpeg files in label"
    flag1 = flag2 = flag3 = 0
    print(f"flag status {flag1} before match")
    file_list = []
    for file in label_file_list:
        tup = []        
        tup.append(file)
        for image1 in green_file_list:            
            if file[:-4] == image1[:-4]: 
                flag1 = 1                   
                tup.append(image1)             
                break    
        for image2 in red_file_list:
            if file[:-4] == image2[:-4]:   
                flag2 = 1     
                tup.append(image2)       
                break
        for image3 in dab_file_list:
            if file[:-4] == image3[:-4]:    
                flag3 =1       
                tup.append(image3) 
                break
        file_list.append(tup)    
       
    if flag1 == flag2 and flag1 == flag3:
        print(f"flag status {flag1} means image pairs matched")   
           
    return file_list

def calculate_features(mask,image):
    gradient_features = htk.features.compute_gradient_features(mask,image)   
    haralick_features = htk.features.compute_haralick_features(mask,image)
    intensity_features = htk.features.compute_intensity_features(mask,image)
    combined_features = pd.concat([gradient_features,haralick_features,intensity_features],axis=1)
    
    chosen_columns = ['Gradient.Canny.Mean','Haralick.Correlation.Mean', 'Haralick.Entropy.Mean',       
       'Intensity.Min','Intensity.Max','Intensity.Median',
       'Intensity.Skewness', 'Intensity.HistEntropy']
    # combined_features = combined_features[chosen_columns]       
    return combined_features

def extract_features(GREEN_PATH=None,RED_PATH=None,DAB_PATH=None, LABEL_IMG=None,**kw):
    img_sets = create_sets(GREEN_PATH,RED_PATH,DAB_PATH, LABEL_IMG)    
    print(f"length image pair {len(img_sets)}")
    common_list = []
    sample = kw["name"]
    for image_tups in img_sets:
        print(f"image tups length {len(image_tups)}")
        df_list = []
        label,green,red,dab = image_tups # these are appended in the same order in the tuple previously
        
        label_img = io.imread(os.path.join(LABEL_IMG,label))       
        if label_img.ndim >2:
            label_img = label_img[:,:,0]
        label_img = remove_small_objects(label_img, min_size=200, connectivity=1) # important-else morphometry fails
        dab_img = io.imread(os.path.join(DAB_PATH,dab)) 
        if dab_img.ndim>2:
            dab_img = dab_img[:,:,0]
        green_img = io.imread(os.path.join(GREEN_PATH,green))
        if green_img.ndim>2:
            green_img = green_img[:,:,0]
        red_img = io.imread(os.path.join(RED_PATH,red))
        if red_img.ndim>2:
            red_img = red_img[:,:,0]
        
        dab_df = calculate_features(label_img,dab_img)
        dab_df_columns = dab_df.columns
        dab_df.columns = ["dab_"+label for label in dab_df_columns]
        green_df = calculate_features(label_img,green_img)
        green_df_columns = green_df.columns
        green_df.columns = ["green_"+label for label in dab_df_columns]
        red_df = calculate_features(label_img,red_img)
        red_df_columns = red_df.columns
        red_df.columns = ["red_"+label for label in dab_df_columns]
        # print(f"red {red_df.shape[0]} dab {dab_df.shape[0]} green {green_df.shape[0]}")
        df_list.append(dab_df)
        df_list.append(green_df)
        df_list.append(red_df)
        # print(f"df list {len(df_list)}")
        df = pd.concat(df_list,axis=1)
        sample_list =[]
        for i in range(df.shape[0]):
            file_head = sample+"_"+label[:-4]+"_"+str(i)
            sample_list.append(file_head)
        df['sample'] = sample_list
        
        # print(f"df shape is {df.shape[0]} and {df.shape[1]}")
        common_list.append(df)
    combined_df = pd.concat(common_list,axis=0)    
    return combined_df


if __name__ == "__main__":
    PATH = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/"
    SET = "set3_core_images_c2d3l64r512/"
    SUBFOLDER = "ub18_66043_7_set3_512px_mask/"
    GREEN_IMG = PATH+SET+SUBFOLDER+"set3_green/"
    RED_IMG = PATH+SET+SUBFOLDER+"set3_red/"
    DAB_IMG = PATH+SET+SUBFOLDER+"set3_dab/"
    LABEL_IMG = PATH+SET+SUBFOLDER+"edited_full_mask_labels/"
    OUT = PATH+SET+SUBFOLDER
    IMG_PATH = ""
    sample = "ub18_66043_7"
      
    stain_features = extract_features(GREEN_PATH=GREEN_IMG,RED_PATH=RED_IMG,DAB_PATH=DAB_IMG, LABEL_IMG=LABEL_IMG,name=sample)
    sample = sample+"_antigen_full.csv"
    stain_features.to_csv(os.path.join(OUT,sample),index=True,index_label="index")