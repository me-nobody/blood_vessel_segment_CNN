/*
Blood Vessel Detector
ImageJ script to detect blood vessels after obtaining the residual image from QuPath
the residual channel min,max and gamma has to be adjusted prior to use
resolution 10
include ROI
apply color transforms
Anubrata Das
*/

output = "C:\\demo_images\\output"

db_vessel(output);
function db_vessel(output){
	//get ID of original image
	original = getImageID();
	// select the original image
	selectImage(original);
	original_name = getTitle();
	
	// create a duplicate image
	run("Duplicate...", "duplicate");
	copy = getImageID();
	selectImage(copy);
	duplicate_name = getTitle();
	//print("duplicated image name is "+duplicate_name);
	name="duplicate_"+original_name;
	rename(name);
	present_image = getTitle();
	//print("renamed duplicated image is "+present_name);

	run("Clear Outside");
	setForegroundColor(0, 0, 0);
	run("Make Inverse");
	run("Fill", "slice");
	run("Make Inverse");
    roiManager("reset");
	run("ROI Manager...");
	
	selectImage(present_image);
	// if run brightness is uncommented then the window remains open
	//run("Brightness/Contrast...");
	run("Enhance Contrast", "saturated=0.35");
	//setAutoThreshold("Default");
	// if run threshold is uncommented then the window remains open
    //run("Threshold...");
    setThreshold(0.05, 1.3);
    setOption("BlackBackground", true);
    run("Convert to Mask");
    run("Set Measurements...", "area mean min display redirect=present_image decimal=2");
    run("Analyze Particles...", "size=10-Infinity show=Outlines display clear include overlay add");
    // save the present Results
    selectWindow("Results");
	saveAs("txt",  output+"/"+"blood_vessel_"+original_name+ ".csv");
	run("Close");
    // save the present ROI
    roiManager("Save", output+"/"+"blood_vessel_"+original_name+"ROIset.zip");
    //roiManager("reset");
   
	// save the blood vessels identified
    drawing_image = "Drawing of "+ present_image;
    //print(drawing_image);
    //
    selectImage(drawing_image);
    saveAs("Jpeg", output +"\\"+ drawing_image);    
    // iterate over the ROIs
    total_roi = roiManager("Count");
	print("total ROI "+total_roi);
	// update the ROI by enlarging the area of selection
	run("Set Measurements...", "area mean min display redirect=present_image decimal=2");
	for (roi=0; roi<total_roi; roi++) { // loop through the rois
        roiManager("Select", roi);
        //print(roi_name);
        roiManager("Rename", "roi_"+roi);
        roi_name = RoiManager.getName(roi);
        run("Enlarge...", "enlarge=4");
        roiManager("Update");
        roiManager("Measure");
        }
	roiManager("Save", output+"/"+"surrounding_area_"+original_name+"ROIset.zip");
	selectImage(present_image);
	roiManager("Show All");
	run("From ROI Manager");
    run("Flatten");
    saveAs("tif", output +"\\"+ present_image);
   
    // save the present Results
    selectWindow("Results");
	saveAs("txt",  output+"/"+"surrounding_area_"+original_name+ ".csv");
	run("Close");
	
	
	// save the updated drawing
	selectImage(drawing_image);
	roiManager("Add", "overlay");
	run("Flatten");
    saveAs("Jpeg", output +"\\"+ "updated_"+drawing_image);
   
    // wait sequence used during development
	  //title = "WaitForUser";
      //msg = "check all results are obtained";
      //waitForUser(title, msg);
	
	
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
	    close("ROI Manager");
	}
 
 
    // close threshold window
	if (isOpen("Brightness/Contrast...")){
	    close("Brightness/Contrast...")
	}
	
	// close threshold window
	
	if (isOpen("Threshold...")){
	    close("Threshold...")
	}
	
    close("*");
	close("\\Others");
	run("Close All");
	//run("Collect Garbage");
	}
