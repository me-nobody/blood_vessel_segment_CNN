"""

33 143 166	   Leading_Edge_LE_(Teal_or_Blue_Areas)
210 5 208	   Infiltrating_Tumor_IT_(Purple_Areas)
5 208 4		   Cellular_Tumor_CT_(Green_Areas)
37 209 247	   Perinecrotic_Zone_CTpnz_(Light_Blue_Areas)
6 208 170	   Pseudopalisading_Cells_Around_Necrosis_CTpan_(Sea_Green_Areas)
255 102 0	   Microvascular_Proliferation_CTmvp_(Red_Areas)
5 5 5		   Necrosis_CTne_(Black_Areas)
255 255 255	   Blood_Cells(White_Areas)

this script is updated to take and interpret the updated zone labels

"""

# Import Required Modules

#import tqdm as tq
import numpy as np

##############################################################################

# Function for converting Integer Values to RGBs
def RGBs_Finder(Encoded_Image, Integer_RGBs):
    R, C = Encoded_Image.shape
    Decoded_Image = np.zeros((R, C, 3), 'uint8')
    for i in range(R):
        for j in range(C):
            Decoded_Image[i, j] = Integer_RGBs[Encoded_Image[i, j]]
    return Decoded_Image


# Function for returning RGB_Values & labels, from Mask_Label.txt
def Text_Inspector(txt):    
    RGB_Values, Labels = txt.strip().split("\t")
    RGB_Values_Tuple = tuple(int(i) for i in RGB_Values.split(" "))
    return RGB_Values_Tuple, Labels


# Function for making two lists i.e. RGB_Values and Labels, and one dictionary containing both RGB values and their corresponding Integers
def RGBs_to_Integers(Mask_Label_Path):    
    RGB_Values_Tuple, Labels_Tuple = zip(*[Text_Inspector(line) for line in open(Mask_Label_Path)])
    RGB_Values_List = list(RGB_Values_Tuple)
    Labels_List = list(Labels_Tuple)
    RGB_and_Integers_Dictionary = {RGB_Values_List[0]:0,RGB_Values_List[1]:1,RGB_Values_List[2]:2,RGB_Values_List[3]:3,RGB_Values_List[4]:4,RGB_Values_List[5]:5,RGB_Values_List[6]:6,RGB_Values_List[7]:7} # Keys = RGBs, Values = Integers e.g.: [5 5 5]:0
    # for updated encoding to reduce the mapping to the 3 key regions, we cannot use enumerate for the color encoding.
    # we have to manually feed the dictionary of the colors
    for color,indexes in RGB_and_Integers_Dictionary.items():
        print(f"keys {color} index {indexes}")
    return RGB_Values_List, Labels_List, RGB_and_Integers_Dictionary
    #return RGB_and_Integers_Dictionary

# Function for RGB values conversion to integers values in each image (Encoding)
def Encoding(Ground_Truths_Index,
             Ground_Truths,
             Out_of_Range_RGB,
             Mask_Width,
             Mask_Height,
             RGB_and_Integers_Dictionary
             ):
    
    Mask_Encoded = np.zeros((Mask_Width, Mask_Height), dtype = "uint8")
    
    for rows in range(Mask_Width):
        for columns in range(Mask_Height):
            
            try: Mask_Encoded[rows, columns] = RGB_and_Integers_Dictionary[tuple(Ground_Truths[Ground_Truths_Index,rows, columns])]
            except: Mask_Encoded[rows, columns] = Out_of_Range_RGB
    return Mask_Encoded


# Main Function
def main(Ground_Truths,
         Mask_Label_Path,
         Mask_Width,
         Mask_Height
         ):
    
    
    Out_of_Range_RGB = 0
    RGB_Values_List, Labels_List, RGB_and_Integers_Dictionary = RGBs_to_Integers(Mask_Label_Path)
    
    print('\n\n\n Please Wait! \n\n Mapping Glioblastoma ground truths to integered encoded masks ... \n')
   
    Mask_Encoded_Final = np.stack([Encoding(Mask_Index, Ground_Truths, Out_of_Range_RGB, Mask_Width, Mask_Height, RGB_and_Integers_Dictionary) for Mask_Index in tq.tqdm(range(len(Ground_Truths)), total = len(Ground_Truths))])               
    return Mask_Encoded_Final                         
    
RGB_and_Integers_Dictionary = RGBs_to_Integers('C:\\Users\\ad394h\\OneDrive - University of Glasgow\\Documents\\Detecting_Tumor_Zones\Deep_learning_CNN_model\\Mask_Labels_updated.txt')    
Integer_to_RGBs = {val: key for (key, val) in RGB_and_Integers_Dictionary.items()}    
for key,value in Integer_to_RGBs.items():
    print(f"value {value} for key {key} in integer2rgb")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




 
