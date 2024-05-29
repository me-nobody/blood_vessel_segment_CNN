
input = "C:\\demo_images\\input"

output = "C:\\demo_images\\output"

list = getFileList(input);
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
	saveAs("Jpeg", output+"/"+present_name);
	//print("renamed duplicated image is "+present_name);
	run("8-bit");
	run("Gaussian Blur...", "sigma=1");
	run("Ridge Detection", "line_width=3.5 high_contrast=230 low_contrast=87 extend_line add_to_manager make_binary method_for_overlap_resolution=SLOPE sigma=1.51 lower_threshold=3.06 upper_threshold=7.99 minimum_line_length=2 maximum=100");
	selectImage(present_name + " Detected segments");
	mask = getTitle();
	saveAs("Jpeg", output+"/"+mask);
	//print("detected mask is "+mask);
	total_roi = roiManager("Count");
	print("total ROI "+total_roi);
	count = 0;
	// roi keeps track of rows in roi manager, while count keeps track of rows in results table
	for (roi=0; roi<total_roi; roi++) { // loop through the rois
    roiManager("Select", roi);
    //print(roi_name);
    roiManager("Rename", "roi_"+roi);
    roi_name = RoiManager.getName(roi);
    run("Set Measurements...", "area mean min display redirect=present_name decimal=3");
    roiManager("Measure");
    // we have to keep track of row number of the results table
    area = getResult("Area", count);
    count = count + 1;
    //print("the area is "+area+" for ROI is "+roi_name);
    //if(area>0){
    if(is("line")){	
    	roiManager("select", roi);
        run("Line to Area");
        run("Enlarge...", "enlarge=2");
        roiManager("Update");
        run("Set Measurements...", "area mean min display redirect=present_name decimal=3");
        roiManager("Measure");
        area = getResult("Area", count);
        //print("the updated area is "+ area+" for ROI is "+ roi_name);   
        count = count+1;
    	}
	}
	
	//roiManager("Save selected", file-path);
	roiManager("Save", output+"/"+"margin_"+original_name+"ROIset.zip");
	// close log
	if (isOpen("Log")) {
	selectWindow("Log");
	run("Close");
	}
	// close results window
    if (isOpen("Results")) {
	selectWindow("Results");
	saveAs("txt",  output+"/"+original_name+ ".csv");
	run("Close");
		}
    // close ROI Manager
    
    close("ROI Manager");

	run("Close All");
	run("Collect Garbage");
	}