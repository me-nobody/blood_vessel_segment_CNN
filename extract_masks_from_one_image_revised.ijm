// Macro to extract a mask from image

//title = "UB19_49455_2_"
//keyword_cd31 = "cd31_region2"
//keyword_claudin = "claudin_region2"
//keyword_gfap = "gfap_region2"
//keyword_he = "he_region2"
//keyword_pdgfrb = "pdgrfb_region2"
//keyword_sma = "sma_region2"
//keyword_laminin = "laminin_region2"

path = "/home/anubratadas/Desktop/"
image = "roi_laminin_region2.tif"
open(path+image);

function processImage() {
    //get ID of original image
    original = getImageID();
    // select the original image
    selectImage(original);
    original_image = getTitle();   
    // get basename of file
    basename = substring(original_image,0,lengthOf(original_image)-4)
    // create a duplicate image
	run("Duplicate...", "duplicate");
	copy = getImageID();
	selectImage(copy);
	duplicate_name = getTitle();	
	name="duplicate_"+original_image;
	rename(name);
	present_image = getTitle();
    basename = substring(present_image,0,lengthOf(present_image)-4)
    print("present_image "+present_image);
    print("basename "+basename);    
    run("Colour Deconvolution", "vectors=[H DAB]");
    selectImage(basename+".tif-(Colour_2)");
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
	name="duplicate_gray";
	rename(name);
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
//  run("Watershed");
    run("Gaussian Blur...", "sigma=2");
//    rename("final_mask");
//    saveAs("Tiff", dir+"/"+"mask_"+filename+".tif");        
//    close("*");
//    close("\\Others");
//    run("Close All");  
        }
processImage();        