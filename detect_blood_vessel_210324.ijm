/*
Blood Vessel Detector
ImageJ script takes RGB images as tif files and then processes them to detect blood vessels
Anubrata Das
*/
input = "C:\\demo_images\\input"

output = "C:\\demo_images\\output"

list = getFileList(input);

setBatchMode(true);

for (i = 0; i < list.length; i++){
     print(list[i]);
     db_vessel(input, output, list[i]);
     wait(20);
}

function db_vessel(input,output,filename){
	open(input+"/"+filename);
	//get ID of original image
	original = getImageID();
	// select the original image
	selectImage(original);
	original_name = getTitle();
	original_name = substring(original_name,0,lengthOf(original_name)-4);
	// create a duplicate image
	run("Duplicate...", "duplicate");
	copy = getImageID();
	selectImage(copy);
	duplicate_name = getTitle();
	//print("duplicated image name is "+duplicate_name);
	name="duplicate_"+original_name;
	rename(name);
	present_name = getTitle();
	
	//print("renamed duplicated image is "+present_name);
	
	run("Colour Deconvolution", "vectors=[H DAB]");
	deconv_image = present_name + "-(Colour_3)";
	selectImage(deconv_image);
	
	run("Brightness/Contrast...");
	run("Enhance Contrast", "saturated=0.35");
	setAutoThreshold("Default");
    run("Threshold...");
    setThreshold(50, 200);
    setOption("BlackBackground", true);
    run("Convert to Mask");
    run("Set Measurements...", "area mean min display redirect=present_name decimal=2");
    run("Analyze Particles...", "size=10-Infinity show=Outlines display clear include overlay add");
     // save the present Results
    selectWindow("Results");
	saveAs("txt",  output+"/"+"blood_vessel_"+original_name+ ".csv");
	run("Close");
    // save the present ROI
    roiManager("Save", output+"/"+"blood_vessel_"+original_name+"ROIset.zip");
    //roiManager("reset");
   
	// save the blood vessels identified
    drawing_image = "Drawing of "+ deconv_image;
    print(drawing_image);
    //
    //selectImage(drawing_image);
    //saveAs("Jpeg", output +"\\"+ drawing_image);    
    // iterate over the ROIs
    total_roi = roiManager("Count");
	print("total ROI "+total_roi);
	// update the ROI by enlarging the area of selection
	run("Set Measurements...", "area mean min display redirect=present_name decimal=2");
	for (roi=0; roi<total_roi; roi++) { // loop through the rois
       roiManager("Select", roi);
       //print(roi_name);
       roiManager("Rename", "roi_"+roi);
       roi_name = RoiManager.getName(roi);
       run("Enlarge...", "enlarge=2");
       roiManager("Update");
       roiManager("Measure");
       
	}
	roiManager("Save", output+"/"+"surrounding_area_"+original_name+"ROIset.zip");
	selectImage(deconv_image);
	roiManager("Show All");
    run("Flatten");
    saveAs("tif", output +"\\"+ deconv_image);
   
     // save the present Results
    selectWindow("Results");
	saveAs("txt",  output+"/"+"surrounding_area_"+original_name+ ".csv");
	run("Close");
	// save the updated drawing
	selectImage(drawing_image);
    saveAs("tif", output +"\\"+ drawing_image);
   

	title = "WaitForUser";
    msg = "If necessary, use the \"Threshold\" tool to\nadjust the threshold, then click \"OK\".";
    waitForUser(title, msg);
	
	
	// close log
	if (isOpen("Log")) {
	selectWindow("Log");
	run("Close");
	}
	// close results window
    if (isOpen("Results")) {
	run("Close");
		}
    // close ROI Manager
    if (isOpen("ROI Manager")) {
	   close("ROI Manager");;
		}
 
    close("*");
	run("Close All");
	run("Collect Garbage");
	}