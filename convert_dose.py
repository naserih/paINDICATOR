import os
import pydicom
import numpy as np
import scipy
from dicom2nrrd import simple_plot_nrrd, get_graph_points, get_rt_structure, convert_rt_structure
import cv2
from PIL import Image, ImageDraw
from skimage import measure
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def get_dose_ref(RT_plan_file):
    pl = pydicom.read_file(RT_plan_file)
    dose_ref = pl.DoseReferenceSequence
    if len(dose_ref) > 1:
        return 'Error: %s dose_refs' %(len(dose_ref)) 
    else:
        return dose_ref[0].DoseReferencePointCoordinates

def convert_dose(RT_dose_file):
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
    

def main():
    patients_dicom_path = "/mnt/iDriveShare/hossein/patientData/TSPINE_cts/"
    case_ID = '0007680'#, # 0035549' # '0007680'
    patient_image_paths = [f for f in os.listdir(patients_dicom_path) if case_ID in f]
    for patient_image_path in patient_image_paths:
        rp_file = [f for f in os.listdir(os.path.join(patients_dicom_path, patient_image_path)) if 'RP.' in f]
        if len(rp_file) > 1:
            print 'ERROR: more than one PLAN found'
        elif len(rp_file) == 0:
            print 'ERROR: No PLAN found'
        rd_file = [f for f in os.listdir(os.path.join(patients_dicom_path, patient_image_path)) if 'RD.' in f]
        if len(rd_file) > 1:
            print 'ERROR: more than one DOSE found'
        elif len(rd_file) == 0:
            print 'ERROR: No DOSE found'
        rs_file = [f for f in os.listdir(os.path.join(patients_dicom_path, patient_image_path)) if 'RS.' in f]
        if len(rp_file) > 1:
            print 'ERROR: more than one RT found'
        elif len(rp_file) == 0:
            print 'ERROR: No PLAN found'
        ct_file = sorted([os.path.join(patients_dicom_path,patient_image_path, f)
                    for f in os.listdir(os.path.join(patients_dicom_path, patient_image_path)) if 'CT.' in f])
        
        # print rd_file
        ds_ref = get_dose_ref(os.path.join(patients_dicom_path,patient_image_path, rp_file[0]))
        ds_array, ds_x,ds_y,ds_z, ds_spacing = convert_dose(os.path.join(patients_dicom_path,patient_image_path, rd_file[0]))
        ct_array, ct_x,ct_y,ct_z, ct_spacing = get_cts(ct_file)

        ds_ref_pxl = (np.argmin(abs(ct_x-ds_ref[0])), np.argmin(abs(ct_y-ds_ref[1])),np.argmin(abs(ct_z-ds_ref[2])))
        # print 'ct_x (%s <> %s)'%(min(ct_x), max(ct_x)), 'ct_y (%s <> %s)'%(min(ct_y), max(ct_y)),'ct_z (%s <> %s)'%(min(ct_z), max(ct_z))
        # print ds_ref, ds_ref_pxl
        contour_names = get_rt_structure(os.path.join(patients_dicom_path,patient_image_path))
        print contour_names
        selected_Contour = 'BONES_RADCAL'
        # selected_Contour = input('[%s] Contour_Names: \t'%rtss_file) 

        rt_mask, pred = convert_rt_structure(os.path.join(patients_dicom_path,patient_image_path),  selected_Contour)
        print ('ct', type(ct_array), ct_array.shape, ct_x.shape, ct_y.shape, ct_z.shape)
        print ('ds', type(ds_array), ds_array.shape, ds_x.shape, ds_y.shape, ds_z.shape)
        # print (ct_x[0],ct_x[1],ct_x[2],'...', ct_x[-1]),(ct_y[0],ct_y[1],ct_y[2],'...',  ct_y[-1]), (ct_z[0], ct_z[1],ct_z[2],'...', ct_z[-1])
        # print (ds_x[0],ds_x[1],ds_x[2],'...' ,ds_x[-1]),(ds_y[0],ds_y[1],ds_y[2],'...', ds_y[-1]), (ds_z[0],ds_z[1],ds_z[2],'...', ds_z[-1])
        # print (ct_x[1]-ct_x[0],ct_x[2]-ct_x[1],ct_x[3]-ct_x[2])
        # print (ct_y[1]-ct_y[0],ct_y[2]-ct_y[1],ct_y[3]-ct_y[2])
        # print (ct_z[1]-ct_z[0],ct_z[2]-ct_z[1],ct_z[3]-ct_z[2])
        # print (ds_x[1]-ds_x[0],ds_x[2]-ds_x[1],ds_x[3]-ds_x[2])
        # print (ds_y[1]-ds_y[0],ds_y[2]-ds_y[1],ds_y[3]-ds_y[2])
        # print (ds_z[1]-ds_z[0],ds_z[2]-ds_z[1],ds_z[3]-ds_z[2])

        ds_mask = map_ds2ct(ds_array, ct_array, ds_x,ds_y,ds_z, ct_x,ct_y,ct_z)
        sphere_50_mask = mask_sphere(ct_array, ct_spacing, ds_ref_pxl, dimameter = 50)
        cubic_50_mask = mask_cubic(ct_array, ct_spacing, ds_ref_pxl, dimameter = 50)
        overlap_mask = binary_overlap_mask(ds_mask, sphere_50_mask, dose_cut=0.9)
        overlap_rt_mask = binary_overlap_mask(rt_mask, sphere_50_mask, dose_cut=0.9)
        # print ds_ref_pxl, ctl_ref_pxl
        # print overlap_mask.shape
        print rt_mask.shape, ds_mask.shape, ct_array.shape
        ctl_ref_pxl = get_graph_points(img=ct_array , xyz_slice = ds_ref_pxl, name = case_ID, spacing = ct_spacing)
        ctl_up_mask = mask_sphere(ct_array, ct_spacing, ctl_ref_pxl[0], dimameter = 50)
        ctl_dn_mask = mask_sphere(ct_array, ct_spacing, ctl_ref_pxl[1], dimameter = 50)
        # print np.amax(ds_mask), np.amax(ds_array)
        slices = [
                    ctl_ref_pxl[1][2]-4, ctl_ref_pxl[1][2], ctl_ref_pxl[1][2]+4
                ]
        cutoff = .9
        # simple_plot_nrrd(img=ds_array, msk=ds_array, sliceNumbers = slices, plotSrc='nrrd')
        # simple_plot_nrrd(img=ds_array, sliceNumbers = slices, plotSrc='nrrd')
        # simple_plot_nrrd(img=ct_array, sliceNumbers = slices, plotSrc='nrrd')
        # simple_plot_nrrd(img=ds_mask  , sliceNumbers = slices, slice_z=ct_z, plotSrc='nrrd')
        # print range(2,5)
        # print 'ds_mask_0', ds_mask_0
        # simple_plot_nrrd(img=ct_array , msk=ctl_dn_mask, ds_0 = ctl_ref_pxl[1], sliceNumbers = slices, dose_cut = cutoff, slice_z=ct_z, plotSrc='nrrd')
        # simple_plot_nrrd(img=ct_array , msk=ctl_dn_mask, ds_0 = ctl_ref_pxl[1], sliceNumbers = slices, dose_cut = cutoff, slice_z=ct_z, plotSrc='nrrd')
        


if __name__ == "__main__":
    main()