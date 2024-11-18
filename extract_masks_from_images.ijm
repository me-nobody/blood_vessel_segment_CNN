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
           run("Duplicate...", " ");
           selectImage(filename+"-1.tif");
           run("Colour Deconvolution", "vectors=[H DAB]");
           selectImage(filename+"-1.tif-(Colour_2)");
           setMinAndMax(100, 255);
           run("Enhance Contrast...", "saturated=0.35 equalize");
           run("Invert");
           run("Grays");
           run("Gaussian Blur...", "sigma=2");
           run("8-bit");
           //run("Kill Borders");
           //run("Threshold...");
           //setAutoThreshold("Default");
           setThreshold(150, 255);
           run("Convert to Mask");
           setOption("BlackBackground", true);
           run("Convert to Mask");
           run("Analyze Particles...", "size=400-100000 circularity=0.00-0.95 show=Masks");
           selectImage("Mask of "+filename+"-1.tif-(Colour_2)");
           //run("Dilate");
//           run("Erode");
//           run("Gaussian Blur...", "sigma=2");
//           run("Invert");
          run("Options...", "iterations=3 count=1 black do=Dilate");
          run("Area Opening", "pixel=200");
          run("Invert");
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

