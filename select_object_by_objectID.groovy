import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathAnnotationObject;

// Get the current project
var currentProject = getProject();

def file1 = new File('/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/image_analysis_Digital_Brain_Tumor/predict75_tumor_object_ids.txt')
def myListOfIds1 = file1.readLines().collect(line -> line.strip())

def file2 = new File('/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/image_analysis_Digital_Brain_Tumor/predict75_normal_object_ids.txt')
def myListOfIds2 = file2.readLines().collect(line -> line.strip())


def cells = getDetectionObjects().findAll{it.isCell()};
println(cells.size())

def tumor_matched = PathObjectTools.findByStringID(myListOfIds1, cells)

def normal_matched = PathObjectTools.findByStringID(myListOfIds2, cells)

resetSelection()

selectObjects(tumor_matched.values())
def tumor = getSelectedObjects()

println("confirmed tumor cells")

println(tumor.size())

def tumorPathClass = getPathClass("Tumor") // Here change to the correct classification!
    tumor.forEach {
    it.setPathClass(tumorPathClass)
}

resetSelection()

selectObjects(normal_matched.values())
def normal = getSelectedObjects()

println("confirmed stroma cells")

println(normal.size())

def normalPathClass = getPathClass("Stroma") // Here change to the correct classification!
    normal.forEach {
    it.setPathClass(normalPathClass)
}

resetSelection()

selectObjects{p -> p.getPathClass() != getPathClass("Tumor") && p.isDetection() && p.getPathClass() != getPathClass("Stroma")}
def other = getSelectedObjects()
println("other cells")

println(other.size())

def otherPathClass = getPathClass("Other") // Here change to the correct classification!
    other.forEach {
    it.setPathClass(otherPathClass)
}

print "Done!"

selectAnnotations()
runPlugin('qupath.lib.plugins.objects.SplitAnnotationsPlugin', '{}');
//runPlugin('qupath.lib.plugins.objects.RefineAnnotationsPlugin', '{"minFragmentSizeMicrons": 2.0,  "maxHoleSizeMicrons": 0.0}');
//saveDetectionMeasurements('/home/anubratadas/Documents/GBM_BBB_LAB_GLASGOW/image_analysis_Digital_Brain_Tumor/object_detection_data/')

def newAnnotations = cells.collect {
    return PathObjects.createAnnotationObject(it.getROI(), it.getPathClass())
}
removeObjects(cells, true)
addObjects(newAnnotations)


