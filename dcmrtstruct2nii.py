# convert_dcmrtstruct2nii.py
import dcmrtstruct2nii 
# import dcmrtstruct2nii, list_rt_structs


def convert(sample_dicom_path, sample_nrrd_path, case_ID):
    print('>>' , sample_dicom_path+case_ID)
    print(dcmrtstruct2nii.list_rt_structs(sample_dicom_path+case_ID))    






if __name__ == '__main__':
    sample_dicom_path = "/mnt/iDriveShare/hossein/2_patientData/IMAGES/"
    sample_nrrd_path = "./nrrd"
    case_ID = '01_anon/rtss'
    convert(sample_dicom_path, sample_nrrd_path, case_ID)