import os
import pydicom
import numpy as np
import scipy
import nrrd
# from dicom2nrrd import simple_plot_nrrd, get_graph_points, get_rt_structure, convert_rt_structure, convert_CT
# from utils.dicom2nrrd import simple_plot_nrrd, get_graph_points, get_rt_structure, convert_rt_structure, convert_CT
from datetime import datetime
import cv2
import csv
import json
from PIL import Image, ImageDraw
from skimage import measure
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# import radiomics_features as rf
import gc
from dotenv import load_dotenv
load_dotenv()

def get_ct_info(CT_files):
    '''
    CT_files: path to the folder containing all CT slices.
    '''
    # print 'ct_files', CT_files
    slices = {} 
    for ct_file in CT_files:
        # print 'ct_file', ct_file
        ds = pydicom.read_file(ct_file)
        # 
        # print image_position
        # print '----------------------------'

        # Check to see if it is an image file.
        # print ds.SOPClassUID
        if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
            #
            # Add the image to the slices dictionary based on its z coordinate position.
            #
            image_position = ds.ImagePositionPatient
            slices[ds.ImagePositionPatient[2]] = ds.pixel_array
            ct_pixel_spacing = ds.PixelSpacing
        else:
            print 'NOOOOOOOOOOOOOOOOOOOOO DS'
            pass



    # The pixel spacing or planar resolution in the x and y directions.
    z = slices.keys()
    z.sort()
    ct_z = np.array(z)
    
        # The ImagePositionPatient tag gives you the x,y,z coordinates of the center of
    # the first pixel. The slices are randomly accessed so we don't know which one
    # we have after looping through the CT slice so we will set the z position after
    # sorting the z coordinates below.
    # print ds
    
    # print 'CT', image_position
    # Construct the z coordinate array from the image index.
    # image_position[2] = ct_z[0]
    # print ds.ConvolutionKernel, ds.FilterType
    # print 

    ct_info = {
        'SamplesPerPixel' : ds.get('SamplesPerPixel'),
        'FilterType': ds.get('FilterType'),
        'ConvolutionKernel': ds.get('ConvolutionKernel'),
        'Exposure': ds.get('Exposure'),
        'XRayTubeCurrent': ds.get('XRayTubeCurrent'),
        'DataCollectionDiameter ': ds.get('DataCollectionDiameter'),
        'KVP': ds.get('KVP'),
        'PixelSpacing': ds.get('PixelSpacing'),
        'SliceThickness': ds.get('SliceThickness'),
        'Manufacturer': ds.get('Manufacturer'),
        'ManufacturerModelName': ds.get('ManufacturerModelName'),
        'StudyDate': ds.get('StudyDate'),
        'PatientSex': ds.get('PatientSex') ,
        'PatientBirthDate': ds.get('PatientBirthDate') ,
        'Rows': ds.get('Rows'),
        'Columns': ds.get('Columns'),
    } 

    # print ds.keys
    # print x
    # Verify z dimension spacing
    b = ct_z[1:] - ct_z[0:-1]
    # z_spacing = 2.5 # Typical spacing for our institution
    if b.min() == b.max():
         z_spacing = b.max()
    else:
        print ('Error z spacing in not uniform')
        z_spacing = 0

    # print z_spacing

    # Append z spacing so you have x,y,z spacing for the array.
    ct_pixel_spacing.append(z_spacing)

    # Build the z ordered 3D CT dataset array.
    ct_array = np.array([slices[i] for i in z])

    # Now construct the coordinate arrays
    # print ct_pixel_spacing, image_position
    x = np.arange(ct_array.shape[2])*ct_pixel_spacing[0] + image_position[0]
    y = np.arange(ct_array.shape[1])*ct_pixel_spacing[1] + image_position[1]
    z = np.arange(ct_array.shape[0])*z_spacing + image_position[2]
    # print x
    # print image_position[0], image_position[1], image_position[2]
    # print ct_pixel_spacing[0], ct_pixel_spacing[1], ct_pixel_spacing[2]
    # print x, y
    # print (len(x), len(y))
    # # The coordinate of the first pixel in the numpy array for the ct is then  (x[0], y[0], z[0])
    return ct_array, x,y,z, ct_pixel_spacing, ct_info

CT_DATABASE_PATH =          '/mnt/iDriveShare/hossein/1_RAWDATA/data/'

ct_folders = os.listdir(CT_DATABASE_PATH)


f_cnt = 0
for ct_folder in ct_folders[f_cnt:]:
    patients_files = os.listdir(os.path.join(CT_DATABASE_PATH, ct_folder))
    p_cnt = 0
    for patient_files in patients_files[p_cnt:]:
        # print os.listdir(os.path.join(CT_DATABASE_PATH, ct_folder, patient_files, 'XY'))
        CT_files = [os.path.join(CT_DATABASE_PATH, ct_folder, patient_files, 'XY', f) for f in os.listdir(os.path.join(CT_DATABASE_PATH, ct_folder, patient_files, 'XY')) if 'CT' in f]
        # print ct_folder, ' >>> ', patient_files
        if len (CT_files) > 0:
            ct_array, x,y,z, ct_pixel_spacing, ct_info = get_ct_info(CT_files)
            print  f_cnt, p_cnt, ct_folder, ',', patient_files, ',', ct_info.keys()
            p_cnt += 1
    f_cnt+= 1
ct_info.keys()
        
      
