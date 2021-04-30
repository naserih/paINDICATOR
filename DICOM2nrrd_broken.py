#convert_dicom2nrrd_2.py
### Python script ###

## Includes ##

import sys
import os
import DICOM

## Script ##

os.system("clear")

print("------------------------------------------------------------")
print("----------------- DICOM to NRRD converter ------------------")
print("------------------------------------------------------------")

# Test list of file presence #

if len(sys.argv)<2:
    print("Error: no file to analyze")
    quit()
else:
    fileName=sys.argv[1]
    print("File analyzed: ",fileName)

# Opening the file #

currentDirectory=os.getcwd() #Get the current directory
pathToInputFile=currentDirectory+"/"+fileName #Get the complete path

inputFile = open(pathToInputFile, "r") #Open the input file in R mode
inputDirectories=inputFile.read() #Read the input file

# Creation of the inputs list #

inputDirectories=inputDirectories.split("\n") #Creation of the list of DICOM directories
inputDirectories=filter(None,inputDirectories) #Empty directories removal

# Processing #

print("------------------------------------------------------------")

if not inputDirectories:
    print("Error: no input to process")
    quit()
else:
    print("List processed: ",inputDirectories)
    
    pathToInputsDirectory=currentDirectory+"/Inputs/"
    pathToOutputsDirectory=currentDirectory+"/Outputs/"
    
    if not os.path.exists(pathToOutputsDirectory):
        os.mkdir(pathToOutputsDirectory)

    for inputDirectory in inputDirectories:

        pathToInputDirectory=pathToInputsDirectory+inputDirectory
        pathToOutputDirectory=pathToOutputsDirectory+inputDirectory

        # Creation of the output directory #

        if not os.path.exists(pathToOutputDirectory):
            os.mkdir(pathToOutputDirectory)
            
            # Slicer processing #
            
            DW=DICOM.DICOMWidget()  
            DSVPC=DICOMScalarVolumePluginClass()
            
            DW.detailsPopup.dicomApp.onImportDirectory(pathToInputDirectory)

            for patient in slicer.dicomDatabase.patients():
                for study in slicer.dicomDatabase.studiesForPatient(patient):
                    for series in slicer.dicomDatabase.seriesForStudy(study):
                        files = slicer.dicomDatabase.filesForSeries(series)
                        if len(files)>1:
                            volumeFile=DSVPC.load(DSVPC.examine([files])[0])
                            slicer.util.saveNode(volumeFile,pathToOutputDirectory=pathToOutputsDirectory+inputDirectory+"/"+inputDirectory+".nrrd")
            
            
        else:
            continue