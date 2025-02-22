import numpy as np
import pandas as pd
import os
import skimage as ski
from skimage import io
from skimage import feature
import histomicstk as htk




def create_pair(GREEN_PATH=None,RED_PATH=None,DAB_PATH=None, LABEL_IMG=None,*label_name):    
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
        for image1 in green_file_list:
            if file[:-4] == image1[:-7]: 
                flag1 = 1   
                tup.append(file)
                tup.append(image1)             
                break    
        for image2 in red_file_list:
            if file[:-4] == image2[:-7]:   
                flag2 = 1     
                tup.append(image2)       
                break
        for image3 in dab_file_list:
            if file[:-4] == image3[:-7]:    
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
    
    chosen_columns = ['Gradient.Mag.Mean', 'Gradient.Mag.Std','Gradient.Canny.Mean','Haralick.Correlation.Mean', 'Haralick.Correlation.Range',       
       'Haralick.Entropy.Mean', 'Haralick.Entropy.Range','Intensity.Min','Intensity.Max','Intensity.Mean', 'Intensity.Median',
       'Intensity.Skewness', 'Intensity.Kurtosis','Intensity.HistEnergy', 'Intensity.HistEntropy']
    combined_features = combined_features[chosen_columns]       
    return combined_features

def extract_features(GREEN_PATH=None,RED_PATH=None,DAB_PATH=None, LABEL_IMG=None,*label_name):
    img_pair = create_pair(GREEN_PATH,RED_PATH,DAB_PATH, LABEL_IMG,*label_name)    
    print(f"length image pair {len(img_pair)}")
    common_list = []
    for pair_list in img_pair:
        df_list = []
        for filename in pair_list:
            if "dd" in filename:
               dab = filename
            elif "gg" in filename:
                green = filename
            elif "rr" in filename:
                red = filename 
            else:
                label = filename    
        label_img = io.imread(os.path.join(LABEL_IMG,label))        
        dab_img = io.imread(os.path.join(DAB_PATH,dab)) 
        green_img = io.imread(os.path.join(GREEN_PATH,green))
        red_img = io.imread(os.path.join(RED_PATH,red))
        
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
        # print(f"df shape is {df.shape[0]} and {df.shape[1]}")
        common_list.append(df)
    combined_df = pd.concat(common_list,axis=0)
    combined_df.to_csv(os.path.join(OUT,filename))
    return combined_df


if __name__ == "__main__":
    GREEN_IMG ="/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/core_images/ub19_52388_5_set1_512px/set1_green/"
    RED_IMG="/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/core_images/ub19_52388_5_set1_512px/set1_red/"
    DAB_IMG="/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/core_images/ub19_52388_5_set1_512px/set1_dab/"
    LABEL_IMG="/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/core_images/ub19_52388_5_set1_512px/BGR2RGB_label_mask/"
    OUT = "/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/multiplex_IHC/core_images/ub19_52388_5_set1_512px/"
    IMG_PATH=""
      
    stain_features = extract_features(GREEN_PATH=GREEN_IMG,RED_PATH=RED_IMG,DAB_PATH=DAB_IMG, LABEL_IMG=LABEL_IMG)
    stain_features.to_csv(os.path.join(OUT,"stain_features.csv"),index=True,index_label="index")