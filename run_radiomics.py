import os
import pydicom
import numpy as np
import scipy
import nrrd
from dicom2nrrd import simple_plot_nrrd, get_graph_points, get_rt_structure, convert_rt_structure, convert_CT
# from utils.dicom2nrrd import simple_plot_nrrd, get_graph_points, get_rt_structure, convert_rt_structure, convert_CT
from datetime import datetime
import cv2
import csv
import json
from PIL import Image, ImageDraw
# from skimage import measure
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
import radiomics_features as rf
import gc
from dotenv import load_dotenv
load_dotenv()
RADIOMICS_FEATURES_PATH_OUT = os.environ.get("RADIOMICS_FEATURES_PATH")

if not os.path.exists(RADIOMICS_FEATURES_PATH_OUT):
        os.makedirs(RADIOMICS_FEATURES_PATH_OUT)

LC_LABEL_LIST = os.environ.get("LESION_CENTERS_WITH_LABEL")
REDO_LC = False # should be false all the time

if REDO_LC:
    print '***************************************************'
    print 'WARNING: REDO_LC should be False for the production'
    print '***************************************************'

def get_dose_ref(RT_plan_file):
    pl = pydicom.read_file(RT_plan_file)
    dose_refs = []
    dose_ref = pl.DoseReferenceSequence
    for dose in dose_ref:
        # print dose.dir()
        if 'DoseReferencePointCoordinates' in dose.dir():
            dose_refs.append(dose.DoseReferencePointCoordinates)
    # print '------>', dose_refs
    return dose_refs

def convert_dose(RT_dose_file):
    '''
    RT_dose_file: single DICOM file of RT DOSE
    '''
    ds = pydicom.read_file(RT_dose_file)
    # print ds.dir()
    dose_pixel = ds.pixel_array
    # print len(ds.PixelData)
    # print dose_pixel.shape

    rows = ds.Rows
    columns = ds.Columns
    pixel_spacing = ds.PixelSpacing
    image_position = ds.ImagePositionPatient
    # print 'DS', image_position
    x = np.arange(columns)*pixel_spacing[0] + image_position[0]
    y = np.arange(rows)*pixel_spacing[1] + image_position[1]
    z = np.array(ds.GridFrameOffsetVector) + image_position[2]
    beam_center = (np.argmin(abs(x)),np.argmin(abs(y)),np.argmin(abs(z)))
    return dose_pixel, x,y,z, pixel_spacing

def get_cts(CT_files):
    '''
    CT_files: path to the folder containing all CT slices.
    '''
    slices = {}
    for ct_file in CT_files:
        ds = pydicom.read_file(ct_file)

        # Check to see if it is an image file.
        # print ds.SOPClassUID
        if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
            #
            # Add the image to the slices dictionary based on its z coordinate position.
            #
            slices[ds.ImagePositionPatient[2]] = ds.pixel_array
        else:
            pass

    # The ImagePositionPatient tag gives you the x,y,z coordinates of the center of
    # the first pixel. The slices are randomly accessed so we don't know which one
    # we have after looping through the CT slice so we will set the z position after
    # sorting the z coordinates below.
    image_position = ds.ImagePositionPatient
    # print 'CT', image_position
    # Construct the z coordinate array from the image index.
    z = slices.keys()
    z.sort()
    ct_z = np.array(z)

    image_position[2] = ct_z[0]

    # The pixel spacing or planar resolution in the x and y directions.
    ct_pixel_spacing = ds.PixelSpacing

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
    return ct_array, x,y,z, ct_pixel_spacing


def map_ds2ct(ds_array, ct_array, ds_x,ds_y,ds_z, ct_x,ct_y,ct_z):
    '''
    ds_array: 3D array of dose [z,y,x] for example dose ds_array[0,:,:] is 2D array 
        of first slice. ds_array[0,0,:] is an array of pixel values of first row in 
        first slice of the dose.
    ct_array: 3D array of CT [z,y,x]
    ds_x: ds_y,ds_z: position of x,y,z components of dose array in mm relative to isocenter. 
        For example ds_x[0] is the position of the first pixel of the ds_array in
        x direction compared to isocenter. like if ds_x[0] = 55, ds_y[0] = -33 and ds_z[0] = 18
        the first pixel of ds_array is in the position of [18,-33,55] mm from isocenter
    ct_x,ct_y,ct_z: osition of x,y,z components of ct array in mm relative to isocenter.
    '''
    ct_z_space = ct_z[1]-ct_z[0]
    ds_z_space = ds_z[1]-ds_z[0]

    ds_0 = (np.argmin(abs(ds_x)),np.argmin(abs(ds_y)),np.argmin(abs(ds_z)))
    # find x and y cut-off for start of image
    if ds_x[0] > ct_x[0]:
        min_ct_x = np.argmin(abs(ct_x-ds_x[0]))
        min_ds_x = 0
    else:
        min_ds_x = np.argmin(abs(ds_x-ct_x[0]))
        min_ct_x = 0 
    
    if ds_x[-1] < ct_x[-1]:
        max_ct_x = len(ct_x)-np.argmin(abs(ct_x-ds_x[-1]))-1
        max_ds_x = len(ds_x)
    else:
        max_ds_x = np.argmin(abs(ds_x-ct_x[-1]))+1
        max_ct_x = 0
        
    # y component of max and min cut_off 
    if ds_y[0] > ct_y[0]:
        min_ct_y = np.argmin(abs(ct_y-ds_y[0]))
        min_ds_y = 0
    else:
        min_ds_y = np.argmin(abs(ds_y-ct_y[0]))
        min_ct_y = 0
         
    if ds_y[-1] < ct_y[-1]:
        max_ct_y = len(ct_y)-np.argmin(abs(ct_y-ds_y[-1]))-1
        max_ds_y = len(ds_y)
    else:
        max_ds_y = np.argmin(abs(ds_y-ct_y[-1]))+1
        max_ct_y = 0
    
    # print min_ds_x, max_ds_x, min_ds_y, max_ds_y
    # print min_ct_x, max_ct_x, min_ct_y, max_ct_y
    
    # print ds_x, ct_x
    # print '-----------------------------------------'
    # print ds_y, ct_y
    # z component of cut off
    if ds_z[0] > ct_z[0]:
        min_ct_z = np.argmin(abs(ct_z-ds_z[0]))
        min_ds_z = 0
    else:
        min_ds_z = np.argmin(abs(ds_z-ct_z[0]))
        min_ct_z = 0 
    if ds_z[-1] < ct_z[-1]:
        max_ct_z = len(ct_z)-np.argmin(abs(ct_z-ds_z[-1]))-1
        max_ds_z = len(ds_z)
    else:
        max_ds_z = np.argmin(abs(ds_z-ct_z[-1]))+1
        max_ct_z = 0
    # print min_ds_z, max_ds_z, min_ct_z, max_ct_z

    # print(ds_z[0], ds_z[-1], ct_z[0], ct_z[-1])
    # print len(ds_z), len(ct_z)

    ct_sampling = None
    ds_sampling = None
    if ct_z_space%ds_z_space == 0:
        ct_sampling = int(ct_z_space/ds_z_space)
    else:
        print'Sampling Error'
    mask = []
    for i in range(0,max_ds_z,ct_sampling):   
        # print i     
        ds_slice = ds_array[i,:,:]
        ct_slice = ct_array[int(i/ct_sampling),:,:]
        trimmed_ds = ds_slice[min_ds_y:max_ds_y, min_ds_x:max_ds_x]
        x_size = ct_slice.shape[1]-min_ct_x-max_ct_x 
        y_size = ct_slice.shape[0]-min_ct_y-max_ct_y 
        '''
        # ## bilinear interpolation Smooth edges but changes the Max values 
        # sampled_ds = scipy.misc.imresize(trimmed_ds, (y_size, x_size),  interp='bilinear', mode='F')
        # SCIPY will not be supported in the future so I am switching to Pillow Image
        ## nn interpolation keeps the max and min the same and replaze the value with NN
        # sampled_ds = np.array(Image.fromarray(trimmed_ds).resize((x_size, y_size), resample=Image.BILINEAR))
        # sampled_ds = np.array(Image.fromarray(trimmed_ds).resize((x_size, y_size), resample=Image.BICUBIC))
        '''
        sampled_ds = np.array(Image.fromarray(trimmed_ds).resize((x_size, y_size), resample=Image.NEAREST))
        # print(np.amax(trimmed_ds), np.amax(sampled_ds))
        paded_ds = np.pad(sampled_ds, ((min_ct_y,max_ct_y), (min_ct_x,max_ct_x)), 'constant') # pad (A, ((top, bottom),(left,right)))
        mask.append(paded_ds)
        # print (ct_slice.shape, ds_slice.shape, trimmed_ds.shape, sampled_ds.shape, paded_ds.shape)
    return np.array(mask)


def mask_sphere(ct_array, ct_spacing, ds_pxl, dimameter = 50):
    z_min = ds_pxl[2]-int(dimameter/ct_spacing[2]/2)
    z_max = ds_pxl[2]+int(dimameter/ct_spacing[2]/2)
    # print x_min, x_max, y_min, y_max, z_min, z_max
    # print ct_array[z_min:z_max,y_min:y_max,x_min:x_max]
    y = np.arange(0, ct_array.shape[1])
    x = np.arange(0, ct_array.shape[2])
    mask = np.zeros((ct_array.shape[0], ct_array.shape[1], ct_array.shape[2]))
    for z_slice in range(z_min,z_max+1,1):
        r2 = (dimameter/2.)**2-((z_slice-ds_pxl[2])*ct_spacing[2])**2
        if r2 >= 0:
            r = np.sqrt(r2)
            # print r2
            mask_slice = np.zeros((ct_array.shape[1], ct_array.shape[2]))
            circle = (x[np.newaxis,:]-ds_pxl[0])**2 + (y[:,np.newaxis]-ds_pxl[1])**2 < r**2
            mask_slice[circle] = 1
            # print ct_array.shape, mask.shape
            # mask[z_min:z_max,y_min:y_max,x_min:x_max] = 1
            mask[z_slice,:,:] = mask_slice
        else:
            print 'Error: r2 is smaller than zero'
    return mask

def mask_cubic(ct_array, ct_spacing, ds_pxl, dimameter = 50):
    x_min = ds_pxl[0]-int(dimameter/ct_spacing[0]/2)
    x_max = ds_pxl[0]+int(dimameter/ct_spacing[0]/2)
    y_min = ds_pxl[1]-int(dimameter/ct_spacing[1]/2)
    y_max = ds_pxl[1]+int(dimameter/ct_spacing[1]/2)
    z_min = ds_pxl[2]-int(dimameter/ct_spacing[2]/2)
    z_max = ds_pxl[2]+int(dimameter/ct_spacing[2]/2)
    
    mask = np.zeros((ct_array.shape[0], ct_array.shape[1], ct_array.shape[2]))
    mask[z_min:z_max,y_min:y_max,x_min:x_max] = 1

    return mask

def mask_cylender(ct_array, ct_spacing, ds_pxl, dimameter = 50, height=50):
    z_min = ds_pxl[2]-int(height/ct_spacing[2]/2)
    z_max = ds_pxl[2]+int(height/ct_spacing[2]/2)
    # print x_min, x_max, y_min, y_max, z_min, z_max
    # print ct_array[z_min:z_max,y_min:y_max,x_min:x_max]
    y = np.arange(0, ct_array.shape[1])
    x = np.arange(0, ct_array.shape[2])
    mask = np.zeros((ct_array.shape[0], ct_array.shape[1], ct_array.shape[2]))
    for z_slice in range(z_min,z_max+1,1):
        r = (dimameter/2.)
        mask_slice = np.zeros((ct_array.shape[1], ct_array.shape[2]))
        circle = (x[np.newaxis,:]-ds_pxl[0])**2 + (y[:,np.newaxis]-ds_pxl[1])**2 < r**2
        mask_slice[circle] = 1
        # print ct_array.shape, mask.shape
        # mask[z_min:z_max,y_min:y_max,x_min:x_max] = 1
        mask[z_slice,:,:] = mask_slice

    return mask

def get_lc_metadata():
    lc_metadata = {}
    ct_list = []
    lc_list = []
    with open (LC_LABEL_LIST, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = csvreader.next()
        for title in header:
            lc_metadata[title] = []

        for row in csvreader:
            ct_list.append(row[1])
            lc_list.append(json.loads(row[12]))
            for i in range(len(row)):
                lc_metadata[header[i]].append(row[i])
    return ct_list, lc_list, lc_metadata



# def get_ref_point(name , database ):
#     pointsource  = "/var/www/devDocuments/hossein/Galenus/data/points/TS_%s_points/%s.txt"%(database, name)
#     ref_point = []
#     with open (pointsource, 'r') as csvfile:
#         csvreader = csv.reader(csvfile)
#         csvreader.next()
#         for row in csvreader:
#             ref_point.append((int(row[0]), int(row[1]),int(row[2])))
#     return ref_point

def  binary_overlap_mask(ds_mask, sphere_mask, dose_cut=0.5):
    max_dose = np.amax(ds_mask)
    dose_bound = dose_cut*max_dose
    binary_mask = np.zeros((ds_mask.shape[0], ds_mask.shape[1], ds_mask.shape[2]))
    for z_slice in range(ds_mask.shape[0]):
        M = np.array(ds_mask[z_slice,:,:], dtype=np.double)
        M[M<dose_bound]= 0
        M[M>=dose_bound]= 1
        overlap = M+sphere_mask[z_slice,:,:]
        overlap[overlap!=2]= 0
        binary_mask[z_slice,:,:]= overlap
 
    return binary_mask
        # if len(contours) > 1:
        #     pass
            # simple_plot_nrrd(img=ds_mask,msk =ds_mask,  ds_0 = ds_pxl, dose_cut = dose_cut , sliceNumbers = [z_slice])
        # print z_slice, len(contours)
    

def process(case_ID, patient_dicom_path, lesion_center, contour, size):
    print patient_dicom_path
    # print g
    print 'case_ID:', case_ID
    print 'patient_dicom_path:', patient_dicom_path
    patient_images_paths = [os.path.join(patient_dicom_path,f) for f in os.listdir(os.path.join(patient_dicom_path))]
    # print('#files', len(patient_images_paths))
    rp_file = [f for f in  patient_images_paths if 'RP.' in f]
    if len(rp_file) > 1:
        print 'ERROR: more than one PLAN found'
    # elif len(rp_file) == 0:
    #     print 'ERROR: No PLAN found'
    rd_file = [f for f in patient_images_paths if 'RD.' in f]
    if len(rd_file) > 1:
        print 'ERROR: more than one DOSE found'
    # elif len(rd_file) == 0:
    #     print 'ERROR: No DOSE found'
    rs_file = [f for f in patient_images_paths if 'RS.' in f]
    if len(rp_file) > 1:
        print 'ERROR: more than one RT found'
    # elif len(rp_file) == 0:
    #     print 'ERROR: No RT structure found'
    ct_file = sorted([f for f in patient_images_paths if 'CT.' in f])
    
    # print('#files', len(patient_images_paths))
    # print('#CTs', len(ct_file))
    # print('#RTs', len(rs_file))
    # print('#PLANs', len(rp_file))
    # print('#DOSEs', len(rd_file))
    ds_refs = []
    if  len(rp_file) > 0:
        # ds_refs = get_dose_ref(os.path.join(patients_dicom_path,'XY',patient_image_path, rp_file[0]))
        pass

    # print 'Dose Ref: ', ds_ref
    if len(rd_file) > 0:
        # ds_array, ds_x,ds_y,ds_z, ds_spacing = convert_dose(os.path.join(patients_dicom_path,patient_image_path, rd_file[0]))
        pass

    ct_array, ct_x,ct_y,ct_z, ct_spacing = get_cts(ct_file)
    # print 'ct_spacing: ', ct_spacing
    ds_ref_pxls = []
    for ds_ref in ds_refs:
        ds_ref_pxls.append((np.argmin(abs(ct_x-ds_ref[0])), np.argmin(abs(ct_y-ds_ref[1])),np.argmin(abs(ct_z-ds_ref[2]))))

    # contour_names = get_rt_structure(os.path.join(patients_dicom_path,patient_image_path))
    # print contour_names
    # selected_Contour = 'BONES_RADCAL'
    # selected_Contour = input('[%s] Contour_Names: \t'%rtss_file) 
    
    # masks = {
    # 'rt_mask' : convert_rt_structure(os.path.join(patients_dicom_path,patient_image_path),  selected_Contour),
    # 'ds_mask' : map_ds2ct(ds_array, ct_array, ds_x,ds_y,ds_z, ct_x,ct_y,ct_z),
    # } 
    # if len(ds_ref_pxls) > 1:
    #     print 'WARNING: len of dose ref pixels is: '%len(ds_ref_pxls)
    
    # if REDO_LC == True:
    #     ctl_ref_pxl = get_ref_point(name = case_ID, database = database)
    # else:
    #     ctl_ref_pxl = get_graph_points(img=ct_array , xyz_slice = ds_ref_pxls[0], 
    #         name = case_ID, spacing = ct_spacing, database = database,  
    #         source= source, size= size)
    # print ctl_ref_pxl
    print 'lesion_center', lesion_center
    if contour == 'SP':
        mask= mask_sphere(ct_array, ct_spacing, lesion_center, dimameter = size)
    elif contour == 'CU':
        mask = mask_cubic(ct_array, ct_spacing, lesion_center, dimameter = size)
    elif contour == 'CY':
        mask = mask_cylender(ct_array, ct_spacing, lesion_center, dimameter = size[0], height=size[1])
        size = '%s:%s'%(size[0],size[1])
    feature_vector = rf.extractor(ct_array, mask, applyLog = False,applyWavelet = False)
    features = {'%s_%s'%(contour, size) : feature_vector}
            

    # overlap_masks = {
    # 'overlap_s50_ds_mask' : binary_overlap_mask(masks['ds_mask'], masks['sphere_50_mask'], dose_cut=0.9),
    # 'overlap_s50_rt_mask' : binary_overlap_mask(masks['rt_mask'], masks['sphere_50_mask'], dose_cut=0.9),
    # }


    # print masks['rt_mask'].shape, masks['ds_mask'].shape, ct_array.shape
    
    # slices = [
    #             ctl_ref_pxl[1][2]-4, ctl_ref_pxl[1][2], ctl_ref_pxl[1][2]+4
    #         ]
    # cutoff = .9

    # simple_plot_nrrd(img=ct_array , msk=ctl_dn_mask, ds_0 = ctl_ref_pxl[1], sliceNumbers = slices, dose_cut = cutoff, slice_z=ct_z, plotSrc='nrrd')
    # simple_plot_nrrd(img=ct_array , msk=ctl_dn_mask, ds_0 = ctl_ref_pxl[1], sliceNumbers = slices, dose_cut = cutoff, slice_z=ct_z, plotSrc='nrrd')
    


        

    return features

def write_csv(csv_path, json_file, case_ID, name_tag):
    with open('%s/%s_%s.csv'%(csv_path,case_ID, name_tag), "wb" ) as csvfile:
        csvwriter = csv.writer(csvfile)
        keys = json_file.keys()
        csvwriter.writerow(['feature']+keys)
        for key in json_file[keys[0]]:
            csvwriter.writerow([key, json_file[keys[0]][key]])
            # csvwriter.writerow([key, json_file[keys[0]][key],json_file[keys[1]][key],json_file[keys[2]][key]])


def remove_failed(radiomics_out):
    failed_files = [os.path.join(radiomics_out, f) for f in os.listdir(radiomics_out) if 'FAILED' in f]
    for failed_file in failed_files:
        os.remove(failed_file)

        
def get_radiomics(contour, size):
    if contour == 'CY':
        if size[0] == size[1]:
            contour_name = '%s%s'%(contour, size[1])
        else:
            contour_name = '%s%s%s'%(contour, size[0], size[1])
    else:
        size = size[0]
        contour_name = '%s%s'%(contour, size)
    cts_path, lcs_list, lc_metadata = get_lc_metadata()
    # print lc_metadata
    print len(cts_path)
    # print g
    case_ids = [f.split('/')[-2] for f in cts_path]
    
    # print case_ids
    # name_tag = '%s_%s'%(database, contour_name)
    # print ct_names
    # patients_dicom_path = "/mnt/iDriveShare/hossein/patientData/TS_%s_cts/"%database
    # already_parsed_path  = "/var/www/devDocuments/hossein/Galenus/data/points/TS_%s_points/"%(database) 
    radiomics_out = os.path.join(RADIOMICS_FEATURES_PATH_OUT, contour_name)
    if not os.path.exists(radiomics_out):
        os.makedirs(radiomics_out)
    remove_failed(radiomics_out)
    already_processed_ids = ['_'.join(f.split('_')[:-1]) for f in os.listdir(radiomics_out) if 'FAILED' not in f]


    # print already_processed_ids
    # print b
    # print already_processed_ids
    # patients_nrrd_path = "/var/www/devDocuments/hossein/Galenus/data/TSPINE_nrrd/"
    # print contour, database, size
    
    # parsed_case_IDs = [f.split(".")[0] for f in os.listdir(already_parsed_path) if '.txt' in f]
    # processed_case_IDs = ["_".join(f.split("_")[0:-1]) for f in os.listdir(already_processed_path) if '.csv' in f]
    # print cts_path
    # case_IDs = [f for f in parsed_case_IDs if f not in processed_case_IDs]
    
    # print 'INFO:  processes Case IDs: %s'%len(case_ids)
    # case_ids = ['0677710']
    feature_metadata = {}

    cnt = 0
    cnt_p = 1
    medians = [0]
    t0 = datetime.now()
    for i in range(len(case_ids)):
        gc.collect()
        features = {}
        cnt += 1
        label_name = '_'.join([case_ids[i]]+[str(f) for f in lcs_list[i]])
        # feature_metadata[case_ids[i]]={}
        if label_name in already_processed_ids or '0_0_0' in label_name:
            print 'Processed'
        else:
            cnt_p += 1
            try:
                t1 = datetime.now()
                features = process(case_ids[i], cts_path[i], lcs_list[i],contour, size)
                write_csv(radiomics_out, features, label_name, contour_name)
                t2 = datetime.now()
                medians.append(((t2-t1).total_seconds())/3600.0)
                # pass
            # print features
            except Exception as e:
                print e
                print 'FAILED'
                features = {'failed':{'failed':'failed'}}
                write_csv(radiomics_out, features, label_name, contour_name+'_FAILED')
        t = datetime.now()
        # print cnt,'/', len(case_ids), ': ', label_name, (contour_name)
        rt = np.median(medians)*(len(case_ids)-cnt)
        if rt > 1:
            print '%s (%i/%i): %5.1f min, Remaning: %8.2f hrs'%(
            contour_name, cnt,len(case_ids), medians[-1]*60, rt)
        else:
            print '%s (%i/%i): %5.1f min, Remaning: %8.2f min'%(
            contour_name, cnt,len(case_ids), medians[-1]*60, rt*60)
        
        # print medians[-1]*60,' min', ''.join(['[']+int(cnt/15)*['|']+int((len(case_ids)-cnt)/15)*[' ']+[']']),int((cnt*100/float(len(case_ids)))),'%', '(', np.median(medians)*(len(case_ids)-cnt),') hrs remaining'
        # feature_metadata[case_ids[i]] = features
    # print feature_vectors

def main():
    # ref_point_source = ['define_manually', 'use_defined_points'][1]
    n = 0
    m = 1
    c = 0
    sizes = [[100,100], [70,70], [50,50], [30,30], [20,20], [15,15], [10,10],
             [30,20], [20,30], [50,30], [30,50]][n:m]
    # sizes = [[50,50]]
    contours = ['CY','SP'][c:c+2]
    # contours = ['CY']
    for size in sizes:
        for contour in contours:
            gc.collect()

            get_radiomics(contour, size)

if __name__ == "__main__":
    main()
