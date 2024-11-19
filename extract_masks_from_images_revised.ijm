// Macro to extract masks from images in directory
//import ij.*;
//import ij.plugin.*;
//import ij.gui.*;
dir = getDirectory("Choose a Directory");
//title = "UB19_49455_2_"
//keyword_cd31 = "cd31_region2"
//keyword_claudin = "claudin_region2"
//keyword_gfap = "gfap_region2"
//keyword_he = "he_region2"
//keyword_pdgfrb = "pdgrfb_region2"
//keyword_sma = "sma_region2"
//keyword_laminin = "laminin_region2"
//print(dir);
// Start recursive function to process files
processDirectory(dir);
function processDirectory(currentDir) {
    list = getFileList(currentDir);    
    for (i = 0; i < list.length; i++) {
        path = currentDir + list[i];   
//        print(path);
        if (endsWith(path,"tif")){
           filename = File.getNameWithoutExtension(path); 
//           print("filename is "+filename);
//           print(path);
           open(path);
           selectImage(filename+".tif");
           base_name = "base_image";
           rename(base_name);
           selectImage(base_name);
           run("Colour Deconvolution", "vectors=[H DAB]");
           selectImage(base_name+"-(Colour_2)");
           im_name = "base_gray";
           rename(im_name);
           setMinAndMax(100, 255);
           run("Enhance Contrast...", "saturated=0.35 equalize");
           run("Invert");
           run("Grays");
           run("Gaussian Blur...", "sigma=2");
           run("8-bit");
           // create a duplicate of the greyed image to create background
           run("Duplicate...", "duplicate");
	      copy = getImageID();
	      selectImage(copy);
	      duplicate_name = getTitle();	
	      dup_name="duplicate_gray";
	      rename(dup_name);
	      duplicate_gray = getTitle();
           //run("Threshold...");
          setThreshold(0, 170);
          setOption("BlackBackground", true);
          run("Convert to Mask");
           // subtract background from grayed image
          imageCalculator("Subtract create", "base_gray","duplicate_gray");
         selectImage("Result of "+im_name);
         sub_name = "subtracted_image";
         rename(sub_name);
         selectImage(sub_name);
         setOption("BlackBackground", true);
         run("Convert to Mask");
         run("Analyze Particles...", "size=400-100000 circularity=0.00-0.95 show=Masks");
         selectImage("Mask of "+sub_name);
         run("Dilate");
         run("Invert");
         run("Area Opening", "pixel=100");         
         run("Gaussian Blur...", "sigma=2");          
         saveAs("Tiff", dir+"/"+"mask_"+filename+".tif");         
         wait(200);          
         close("*");
         close("\\Others");
         run("Close All");  
         close(path);
        }
        else {
             print("path not recognized");
  }
}

