
//get ID of original image
original = getImageID();
// select the original image
selectImage(original);
original_name = getTitle();
//print(original_name);
run("Colour Deconvolution", "vectors=[H DAB]");
selectImage(original_name + "-(Colour_2)");
dab_channel_name = getTitle();
//print(dab_channel_name);
//run("Brightness/Contrast...");
setMinAndMax(120, 180);
setOption("BlackBackground", true);
run("Convert to Mask");
//run("Close All");
