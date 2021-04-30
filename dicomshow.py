import os
from dicom_utils import get_cts

folder_path = r'C:\Users\FEPC-L389\Google Drive\1_PhDProject\Galenus\DICOMfortable\static\data\07-02-2003-p4-14571'
patient_image_path = r'/1.000000-P4P100S300I00003 Gated 0.0A-29193'
ct_file = [os.path.join(folder_path+patient_image_path, f) for f in os.listdir(folder_path+patient_image_path)]
# print (ct_file)
ct_array, ct_array_hu, ct_x,ct_y,ct_z, ct_spacing = get_cts(ct_file)
print ct_array.shape

print ct_array[:,1,:]