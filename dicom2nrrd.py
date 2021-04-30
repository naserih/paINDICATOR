'''
Date: 2020-01-23
Author: Hossein Naseri
Affliation: McGill University Health Center
this script is to convert DICOM CT images and RT structures to nrrd to be used
in pydicom script
'''
import os
import subprocess
import nrrd
from dicom_compiler import Dicom_to_Imagestack
# from dicomDOSE_compiler import Dicom_to_Imagestack
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
# from plot_nrrd import plot_nrrd
import SimpleITK as sitk

def get_rt_structure(rtss_file_path):
    Dicom_reader = Dicom_to_Imagestack(get_images_mask=False)
    Dicom_reader.down_folder(rtss_file_path)
    # See all rois in the folders
    return Dicom_reader.all_s_rois

def convert_rt_structure(path, selected_Contour):
    # print(Dicom_reader.all_s_rois[6], unicode('PTV_800'))
    # Dicom_reader = Dicom_to_Imagestack(get_images_mask=False) 
    print (selected_Contour)
    Dicom_reader = Dicom_to_Imagestack(
        get_images_mask=True, 
        Contour_Names=[selected_Contour],
        )
    print('---------------------------------------------------')
    Dicom_reader.Make_Contour_From_directory(path)
    image = Dicom_reader.ArrayDicom
    mask = Dicom_reader.mask
    # pred = np.zeros([mask.shape[2],mask.shape[1],mask.shape[0]]) # prediction needs to be [# images, rows, cols, # classes]
    # for i in range(mask.shape[0]):
    #     pred[:,:,i] = mask[i,:,:]
    return mask

def write_single_dicom(Dicom_reader, pred):
    Dicom_reader.with_annotations(pred,output_path,ROI_Names=['test'])

def write2nrrd(mask, output_file_path):
    # Write to a NRRD file
    nrrd.write(output_file_path, mask)

def get_graph_points(img=[], xyz_slice=None, spacing = (1,1,1), name='NaN', database = None, source= None, size= None):
    # get set slice position to be in the middle if it is n0t provided
    show_original = False
    if xyz_slice[0] == None:
        x_slice = int(img.shape[2]/2)
    else:
        x_slice = xyz_slice[0] 
    if xyz_slice[1] == None:
        y_slice = int(img.shape[1]/2)
    else:
        y_slice = xyz_slice[1] 
    if xyz_slice[2] == None:
        z_slice = 0 
    else:
        z_slice = xyz_slice[2] 
    # find aspect ration in xyz firections. Spacings could be diffrent in various directions
    xy_ratio = spacing[0]/spacing[1]
    yz_ratio = spacing[1]/spacing[2]
    xz_ratio = spacing[0]/spacing[2]

    # 2D image, single slice at z position
    XY = img[z_slice, :,:]
    ''' 
    choice is satisfaction index of the selected point. You select
    one/multiple ROI point(s) in the image and if you are satisfied with that/those point(s) 
    you put "y" and it moves to the next image. if you select "n" program will bring 
    the same image again and you will sellect point(s) again 
    '''
    choice = 'n'
    while choice != 'y':
        # plots XY plot (slice). move origin to lower corner
        fg, ax = plt.subplots(1)
        fg.suptitle('Click on the T spine bone center')
        ax.grid(True)
        ax.imshow(XY, cmap=plt.cm.gray, origin='lower')
        
        ax.plot(x_slice, y_slice, 'r+') 
        ax.set_aspect(1/xy_ratio)
        # plot ellipse or size 50mm around the point of interest
        ellipse = Ellipse(xy=(x_slice, y_slice), width=50/spacing[0], height=50/spacing[1], 
                            edgecolor='r', fc='None', lw=1)
        ax.add_patch(ellipse)
        # get point of interest from image by clicking. to get 'm' points use ginput(m)
        testPoint_xy = plt.ginput()

        if len(testPoint_xy) > 0:
            testPoint_xy = testPoint_xy[0]
            plt.close('all')
            # crop box of 8x8x4 cm around point of interest 
            y_margin = 80
            x_margin = 80
            z_margin = 40
            if int(testPoint_xy[0]) < x_margin:
                x_margin = int(testPoint_xy[0])
            if int(testPoint_xy[1]) < y_margin:
                y_margin = int(testPoint_xy[1])
            if int(z_slice) < z_margin:
                z_margin = int(z_slice)
            x_0 = int(testPoint_xy[0])-x_margin
            y_0 = int(testPoint_xy[1])-y_margin
            z_0 = int(z_slice)-z_margin
            
            print(testPoint_xy)
            YZ = img[z_0:z_0+2*z_margin,y_0:y_0+2*y_margin,int(testPoint_xy[0])]
            XZ = img[z_0:z_0+2*z_margin,int(testPoint_xy[1]),x_0:x_0+2*x_margin]

            fig, axs = plt.subplots(1,2)
            fig.suptitle('p:%s'%(name))

            axs[0].grid(True)
            axs[0].imshow(YZ, cmap=plt.cm.gray, origin='lower')
            axs[0].set_aspect(1/yz_ratio)
            if show_original:
                axs[0].plot(y_slice-y_0, z_slice-z_0, 'r+') 
                ellipse = Ellipse(xy=(y_slice-y_0, z_slice-z_0), width=50/spacing[1], height=50/spacing[2], 
                                    edgecolor='r', fc='None', lw=0.5)
                axs[0].add_patch(ellipse)
            testPoint_yz = plt.ginput()
            if len(testPoint_yz) > 0:
                for point in testPoint_yz:
                    axs[0].plot(point[0],point[1], 'g+')
                    ellipse = Ellipse(xy=(point[0],point[1]), width=50/spacing[1], height=50/spacing[2], 
                                    edgecolor='r', fc='None', lw=1)
                    # plot 5x5 cm rectangle around point of interest
                    rect = Rectangle(xy=(point[0]-50/spacing[1]/2,point[1]-50/spacing[2]/2), width=50/spacing[1], height=50/spacing[2], 
                                    edgecolor='r', fc='None', lw=1)
                    if source == 'SP':
                        axs[0].add_patch(ellipse)
                    else:
                        axs[0].add_patch(rect)

                axs[1].grid(True)
                axs[1].imshow(XZ, cmap=plt.cm.gray,  origin='lower')
                axs[1].set_aspect(1/xz_ratio)
                if show_original:
                    axs[1].plot(x_slice-x_0, z_slice-z_0, 'r+')
                    ellipse = Ellipse(xy=(x_slice-x_0, z_slice-z_0), width=50/spacing[0], height=50/spacing[2], 
                                        edgecolor='r', fc='None', lw=0.5)
                    
                    axs[1].add_patch(ellipse)
                    
                testPoint_xz = plt.ginput()
                if len(testPoint_xz) > 0:
                    for i in range(len(testPoint_xz)):
                        axs[1].plot(testPoint_xz[i][0],testPoint_yz[i][1], 'g+')
                        ellipse = Ellipse(xy=(testPoint_xz[i][0],testPoint_yz[i][1]), width=50/spacing[0], height=50/spacing[2], 
                                        edgecolor='r', fc='None', lw=1)
                        rect = Rectangle(xy=(testPoint_xz[i][0]-50/spacing[0]/2,testPoint_yz[i][1]-50/spacing[2]/2), 
                                        width=50/spacing[0], height=50/spacing[2], 
                                        edgecolor='r', fc='None', lw=1)
                        if source == 'SP':
                            axs[1].add_patch(ellipse)
                        else:
                            axs[1].add_patch(rect)


                    plt.savefig("/var/www/devDocuments/hossein/Galenus/data/points/TS_%s_points/%s.jpg"%(database, name))
                    plt.close()
                choice = raw_input('Satisfied? ').lower() # accept both Y and y as yes.
    control_refs = []
    with open("/var/www/devDocuments/hossein/Galenus/data/points/TS_%s_points/%s.txt"%(database, name), "w") as text_file:
        text_file.write('testPoint_x,testPoint_y, testPoint_z \n')
        for i in range(len(testPoint_yz)):
            control_refs.append((int(testPoint_xz[i][0]), int(testPoint_yz[i][0]), int(testPoint_yz[i][1])))
            text_file.write('%s,%s,%s \n'%(int(testPoint_xz[i][0]), int(testPoint_yz[i][0]), int(testPoint_yz[i][1])))
    return control_refs


def simple_plot_nrrd(img=[], msk=None, sliceNumbers = None, ds_0 = None, dose_cut=None, slice_z=None, plotSrc='nrrd'):
    if isinstance(sliceNumbers, int):
        sliceNumbers = [sliceNumbers]
    fig = plt.figure()
    if plotSrc == 'sitk':
        img = img.transpose()
        if msk is not None:
            msk = msk.transpose()

    if not sliceNumbers:
        sliceNumbers = [int(img.shape[0]/2)]
    # print(sliceNumbers)
    for sliceNumber in sliceNumbers:
        if slice_z is not None:
            label = slice_z[sliceNumber]
        else:
            label = '_'
        H = img[sliceNumber,:,:]
        plt.imshow(H, cmap=plt.cm.gray)
        plt.title('slice:%s_z:%s'%(sliceNumber,label))
        if msk is not None:
            D = msk[sliceNumber,:,:]
            M = np.array(msk[sliceNumber,:,:], dtype=np.double)
            # M[M != 50] = np.nan
            max_dose = np.amax(msk)
            dose_bound = dose_cut*max_dose
            M[M<dose_bound]= np.nan
            M[M>=dose_bound]= 1
            S = M.sum()
            if S != 0:
                plt.imshow(M, alpha=0.5,  vmin = 0)
                if ds_0:
                    if  ds_0[2] == sliceNumber:
                        plt.plot(ds_0[0], ds_0[1], 'r+')
                    else:
                        plt.plot(ds_0[0], ds_0[1], 'b+')
                plt.title('slice:%s_z:%s_Dose:%s=%s'%(sliceNumber,label,dose_cut, dose_bound))
                # plt.imshow(D, alpha=0.5,  cmap=plt.cm.hot)
                # plt.title('slice:%s_z:%s_Dose:%s=%s'%(sliceNumber,label,dose_cut, dose_bound))
            else:
                nonEmptySlices = {}
                for i in range(img.shape[0]):
                    M = msk[i,:,:]
                    S = M.sum()
                    if S != 0:
                        nonEmptySlices[i]=S
                        print i, S
                print('Empty slice!')
        plt.show()

def convert_CT(patients_dicom_path, patients_nrrd_path, case_ID):
    if case_ID == 'all':
         patients_dicom_folders = os.listdir(patients_dicom_path)
    else:
        patients_dicom_folders = [f for f in os.listdir(patients_dicom_path) if case_ID in f]    
    print(patients_dicom_folders)
    plastimatch = '/usr/bin/plastimatch'
    for dicom_file in patients_dicom_folders:
        nrrd_file = os.path.join(patients_nrrd_path, dicom_file)
        if not os.path.exists(nrrd_file):
            os.mkdir(nrrd_file)
        print(nrrd_file)
        if 'ct_image.nrrd' not in os.listdir(nrrd_file):
            print("-------------------")
            plastimatch_img_arg = [
                'plastimatch', 'convert', 
                '--input',os.path.join(patients_dicom_path, dicom_file), 
                '--output-img', '%s/ct_image.nrrd'%(nrrd_file)
            ]
            plastimatch_rt_arg = [
                'plastimatch', 'convert', 
                '--input-ss-img',os.path.join(patients_dicom_path,dicom_file), 
                # '--input-prefix', 'dcm',
                '--output-ss-img', '%s/segment_rtss.nrrd'%(nrrd_file),
                '--output-ss-list', '%s/segment_rtss.txt'%(nrrd_file), 
                # '--prefix-format', "nrrd",
                '--output-prefix', nrrd_file,
                '--referenced-ct', os.path.join(patients_dicom_path,dicom_file),
                '--output-labelmap', '%s/label'%(nrrd_file),
                        ]
             # print(plastimatch_img_arg)
            # print(plastimatch_rt_arg)
            print("-----------------")
            subprocess.call(plastimatch_img_arg)
            print("--CT CONVERTED--%s"%dicom_file)
            # subprocess.call(plastimatch_rt_arg)
            # print("--RT CONVERTED--%s"%patients_dicom_folders)


def main():
    check_chache = False
    patients_dicom_path = "/mnt/iDriveShare/hossein/patientData/TSPINE_cts/"
    # sample_dicom_path = "./"
    patients_nrrd_path = "/var/www/devDocuments/hossein/Galenus/data/TSPINE_nrrd/"
    if not os.path.exists(patients_nrrd_path):
            os.mkdir(patients_nrrd_path)

    # case_ID = 'all'
    case_ID = '0374317' #0196337 #'0007680'
    convert_CT(patients_dicom_path, patients_nrrd_path, case_ID)
    print("DONE!%s"%(1))

    ### very very important file path sould be absolutely without / at the end!
    if case_ID == 'all':
        rtss_paths = os.listdir(patients_dicom_path)
    else:
        rtss_paths = [f for f in os.listdir(patients_dicom_path) if case_ID in f]    
    print rtss_paths
    for rtss_file in rtss_paths:
        rtss_file_path = os.path.join(patients_dicom_path,rtss_file)
        Contour_Names = get_rt_structure(rtss_file_path)
        # print(os.listdir(os.path.join(patients_nrrd_path,rtss_file)))
        if 'mask.nrrd' in os.listdir(os.path.join(patients_nrrd_path,rtss_file)) and check_chache == True:
            print("mask is already here")
            continue
        print(Contour_Names)
        # print("DONE!%s"%(2))
        if len(Contour_Names) != 0 :
            selected_Contour = 'BONES_RADCAL'
            # selected_Contour = input('[%s] Contour_Names: \t'%rtss_file) 

            mask = convert_rt_structure(os.path.join(patients_dicom_path,rtss_file),  selected_Contour)
            print("--RT CONVERTED--%s"%rtss_file)
            output_file_path = '%s/%s/rtss_mask_%s.nrrd'%(patients_nrrd_path, rtss_file, selected_Contour)
            write2nrrd(mask, output_file_path)
            output_file_path_noname = '%s/%s/mask.nrrd'%(patients_nrrd_path, rtss_file)
            write2nrrd(mask, output_file_path_noname)
            read_ss_data, ss_header = nrrd.read(output_file_path)
            print("--RT SAVED--%s"%rtss_file)
            output_file_path = '%s/%s/ct_image.nrrd'%(patients_nrrd_path, rtss_file)
            read_ct_data, ct_header = nrrd.read(output_file_path)
            write2nrrd(read_ct_data.transpose(), '%s/%s/image.nrrd'%(patients_nrrd_path, rtss_file))
            print("--CT SAVED--%s"%rtss_file)
            image = read_ct_data.transpose()
            # print(ss_header)
            # simple_plot_nrrd(img=image, msk=mask, sliceNumber = None, plotSrc=['nrrd','sitk'][0])
            print("DONE! CT: %s RT: %s IMAGE: %s MASK: %s"%(read_ct_data.shape, read_ss_data.shape, image.shape, mask.shape))

        else:
            print("ERROR: Countours are unkown for %s" %os.path.join(patients_dicom_path,rtss_file))

if __name__ == '__main__':
    main()