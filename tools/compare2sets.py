import os
originalPath = '../../DATA/patients_documents/SPINE_MET_pdfs'

oldPath = '../../DATA/patients_documents/TS_MET_pdfs'
newSet = os.listdir(originalPath)
allOld = [os.listdir(os.path.join(oldPath,f)) for f in os.listdir(oldPath)]
# print len([item for sublist in allOld for item in sublist])
flat_list = list(set([item for sublist in allOld for item in sublist]))
print len(flat_list)
print len(newSet)
