#radiomics_features.py
from __future__ import print_function
import importlib
from distutils.version import LooseVersion
import SimpleITK as sitk
from plot_nrrd import plot_nrrd
import nrrd
from dicom2nrrd import simple_plot_nrrd
from radiomics import featureextractor, firstorder, getTestCase, glcm, glrlm, glszm, ngtdm,gldm,  imageoperations, shape
import matplotlib.pyplot as plt
import numpy as np
import six
import os
import dicom2nrrd
import csv
import random 
from dotenv import load_dotenv
load_dotenv()

DATA_CACHE_PATH_OUT = os.environ.get("DATA_CACHE_PATH")
if not os.path.exists(DATA_CACHE_PATH_OUT):
    os.makedirs(DATA_CACHE_PATH_OUT)


def check_requirements():
    # check that all packages are installed (see requirements.txt file)
    required_packages = {'jupyter', 
                         'numpy',
                         'matplotlib',
                         'ipywidgets',
                         'scipy',
                         'pandas',
                         'SimpleITK',
                         'radiomics',
                         'pydicom'
                        }

    problem_packages = list()
    # Iterate over the required packages: If the package is not installed
    # ignore the exception. 
    for package in required_packages:
        try:
            p = importlib.import_module(package)        
        except ImportError:
            problem_packages.append(package)
        
    if len(problem_packages) is 0:
        print('All is well.')
    else:
        print('The following packages are required but not installed: '+
                ', '.join(problem_packages))



def old_radiomics():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    # Enable/Disable all or selected feature classes
    # extractor.enableAllFeatures()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('firstorder')
    # extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
    # extractor.enableAllImageTypes()
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName('Original') 
    # extractor.enableImageTypeByName('Wavelet') 

    print('Extraction parameters:\n\t', extractor.settings)
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    print('Enabled features:\n\t', extractor.enabledFeatures)

    feature_vectors = {}
    cnt = 0
    for case_id in valid_IDs:
        cnt += 1
        # try:
        ct_nrrd_path = os.path.join(patients_nrrd_path,case_id, "image.nrrd")
        ss_nrrd_path = os.path.join(patients_nrrd_path,case_id, "rtss_mask_GTV_800.nrrd")
        print("Reading ct image")
        image = sitk.ReadImage(ct_nrrd_path)
        # image, header = nrrd.read(ct_nrrd_path)
        print("Reading roi mask")
        mask = sitk.ReadImage(ss_nrrd_path)
        # mask, header = nrrd.read(ss_nrrd_path)
        print("Getting ct image array")
        image_array = sitk.GetArrayFromImage(image)
        print("Getting roi mask array")
        mask_array = sitk.GetArrayFromImage(mask)
        print(image_array.shape, mask_array.shape)
        # simple_plot_nrrd(image_array, mask_array, sliceNumber=75, plotSrc='sitk')
        print (cnt, "_ Calculating features: ",case_id)
        feature_vectors[case_id] = extractor.execute (image, mask)
        # except exception as e:
            # print('Error: No Image/Mask', os.path.join(patients_nrrd_path,case_id)
            # print (e)
    feature_names = list(sorted(filter(lambda k: k.startswith("original_"), feature_vectors[case_id])))
    print(feature_names)
    print ('DONE! %s Features extracted' %len(feature_names))

    feature_matrix = np.zeros((len(feature_vectors),len(feature_names)))

    for i in range(len(valid_IDs)):
    # for i in range(11):
        a = np.array([])
        for feature_name in feature_names:
            a = np.append(a, feature_vectors[valid_IDs[i]][feature_name])
        feature_matrix[i,:] = a
    # May have NaNs
    feature_matrix = np.nan_to_num(feature_matrix)
    print(feature_matrix.shape)



    for f in extractor.enabledFeatures.keys():
      print('  ', f)
      print(getattr(extractor, 'get%sFeatureValue' % f).__doc__)

    print('Calculating first order features...')
    results = extractor.execute()
    print('done')

    print('Calculated first order features: ')
    for (key, val) in six.iteritems(results):
      print('  ', key, ':', val)



    print('Active features:')

    for cls, features in six.iteritems(extractor.enabledFeatures):
        if len(features) == 0:
            features = [f for f, deprecated in six.iteritems(extractor.getFeatureNames(cls)) if not deprecated]
        for f in features:
            print(f)
            print(getattr(extractor.featureClasses[cls], 'get%sFeatureValue' % f).__doc__)

    mask = None
    image = None
    feature_matrix = None
    feature_vectors = None
    extractor = None
    mask_array = None
    image_array - None

def features_extractor(patients_nrrd_path, valid_IDs, applyLog = False, applyWavelet = False):
    feature_vectors = {}
    cnt = 0
    for case_id in valid_IDs:
        feature_vectors[case_id] = {}
        cnt += 1
        # try:
        ct_nrrd_path = os.path.join(patients_nrrd_path,case_id, "image.nrrd")
        ss_nrrd_path = os.path.join(patients_nrrd_path,case_id, "mask.nrrd")
        print("Reading ct image")
        image = sitk.ReadImage(ct_nrrd_path)
        # image, header = nrrd.read(ct_nrrd_path)
        print("Reading roi mask")
        mask = sitk.ReadImage(ss_nrrd_path)
        # mask, header = nrrd.read(ss_nrrd_path)
        print("Getting ct image array")
        image_array = sitk.GetArrayFromImage(image)
        print("Getting roi mask array")
        mask_array = sitk.GetArrayFromImage(mask)
        print(image_array.shape, mask_array.shape)
        # simple_plot_nrrd(image_array, mask_array, sliceNumber=75, plotSrc='sitk')
        print (cnt, "_ Calculating features: ",case_id)   

        settings = {'binWidth': 25,
                'interpolator': sitk.sitkBSpline,
                'resampledPixelSpacing': None}
        interpolator = settings.get('interpolator')
        resampledPixelSpacing = settings.get('resampledPixelSpacing')
        if interpolator is not None and resampledPixelSpacing is not None:
            image, mask = imageoperations.resampleImage(image, mask, **settings)
        bb, correctedMask = imageoperations.checkMask(image, mask)
        if correctedMask is not None:
          mask = correctedMask
        image, mask = imageoperations.cropToTumorMask(image, mask, bb)
        
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
        # firstOrderFeatures.enableFeatureByName('Mean', True)
        firstOrderFeatures.enableAllFeatures()

        # print('Will calculate the following first order features: ')
        # for f in firstOrderFeatures.enabledFeatures.keys():
        #   print('  ', f)
        #   print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)

        # print('Calculating first order features...')
        results = firstOrderFeatures.execute()
        # print('done')

        print('Calculated first order features: ')
        for (key, val) in six.iteritems(results):
            firstOrderFeatureName = '%s_%s' % ('firstOrder', key)
            if firstOrderFeatureName not in feature_vectors[case_id]:
                feature_vectors[case_id][firstOrderFeatureName] = val
            else:
                print('Error: firstOrder key existing! %s'%firstOrderFeatureName)
                # break
            # print('  ', key, ':', val)

        #
        # Show Shape features
        #

        shapeFeatures = shape.RadiomicsShape(image, mask, **settings)
        shapeFeatures.enableAllFeatures()

        # print('Will calculate the following Shape features: ')
        # for f in shapeFeatures.enabledFeatures.keys():
        #   print('  ', f)
        #   print(getattr(shapeFeatures, 'get%sFeatureValue' % f).__doc__)

        # print('Calculating Shape features...')
        results = shapeFeatures.execute()
        # print('done')

        print('Calculated Shape features: ')
        for (key, val) in six.iteritems(results):
            ShapeFeatureName = '%s_%s' % ('Shape', key)
            if ShapeFeatureName not in feature_vectors[case_id]:
                feature_vectors[case_id][ShapeFeatureName] = val
            else:
                print('Error: shape key existing! %s'%ShapeFeatureName)
                # break
            # print('  ', key, ':', val)

        #
        # Show GLCM features: Gray Level Co-occurrence Matrix (GLCM) Features
        #
        glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
        glcmFeatures.enableAllFeatures()

        # print('Will calculate the following GLCM features: ')
        # for f in glcmFeatures.enabledFeatures.keys():
        #   print('  ', f)
        #   print(getattr(glcmFeatures, 'get%sFeatureValue' % f).__doc__)

        # print('Calculating GLCM features...')
        results = glcmFeatures.execute()
        # print('done')

        print('Calculated GLCM features: ')
        for (key, val) in six.iteritems(results):
            GLCMFeatureName = '%s_%s' % ('GLCM', key)
            if GLCMFeatureName not in feature_vectors[case_id]:
                feature_vectors[case_id][GLCMFeatureName] = val
            else:
                print('Error: GLCM key existing! %s'%GLCMFeatureName)
                # break
            # print('  ', key, ':', val)



        #
        # Show GLSZM features; Gray Level Size Zone Matrix (GLSZM) Features
        #
        glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
        glszmFeatures.enableAllFeatures()

        # print('Will calculate the following GLSZM features: ')
        # for f in glszmFeatures.enabledFeatures.keys():
        #   print('  ', f)
        #   print(getattr(glszmFeatures, 'get%sFeatureValue' % f).__doc__)

        # print('Calculating GLSZM features...')
        results = glszmFeatures.execute()
        print('done')

        print('Calculated GLSZM features: ')
        for (key, val) in six.iteritems(results):
            GLSZMFeatureName = '%s_%s' % ('GLSZM', key)
            if GLSZMFeatureName not in feature_vectors[case_id]:
                feature_vectors[case_id][GLSZMFeatureName] = val
            else:
                print('Error: GLSZM key existing! %s'%GLSZMFeatureName)
                # break
            # print('  ', key, ':', val)


        #
        # Show GLRLM features; Gray Level Run Length Matrix (GLRLM) Features
        #
        glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
        glrlmFeatures.enableAllFeatures()

        # print('Will calculate the following GLRLM features: ')
        # for f in glrlmFeatures.enabledFeatures.keys():
        #   print('  ', f)
        #   print(getattr(glrlmFeatures, 'get%sFeatureValue' % f).__doc__)

        # print('Calculating GLRLM features...')
        results = glrlmFeatures.execute()
        # print('done')

        print('Calculated GLRLM features: ')
        for (key, val) in six.iteritems(results):
            GLRLMFeatureName = '%s_%s' % ('GLRLM', key)
            if GLRLMFeatureName not in feature_vectors[case_id]:
                feature_vectors[case_id][GLRLMFeatureName] = val
            else:
                print('Error: GLRLM key existing! %s'%GLRLMFeatureName)
                # break
            # print('  ', key, ':', val)

        #
        # Show NGTDM features; Neighbouring Gray Tone Difference Matrix (NGTDM) Features
        #
        ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask, **settings)
        ngtdmFeatures.enableAllFeatures()

        # print('Will calculate the following NGTDM features: ')
        # for f in ngtdmFeatures.enabledFeatures.keys():
        #   print('  ', f)
        #   print(getattr(ngtdmFeatures, 'get%sFeatureValue' % f).__doc__)

        # print('Calculating NGTDM features...')
        results = ngtdmFeatures.execute()
        # print('done')

        print('Calculated NGTDM features: ')
        for (key, val) in six.iteritems(results):
            NGTDMFeatureName = '%s_%s' % ('NGTDM', key)
            if NGTDMFeatureName not in feature_vectors[case_id]:
                feature_vectors[case_id][NGTDMFeatureName] = val
            else:
                print('Error: NGTDM key existing! %s'%NGTDMFeatureName)
                # break
            # print('  ', key, ':', val)

        #
        # Show GLDM features; Gray Level Dependence Matrix (GLDM) Features
        #
        gldmFeatures = gldm.RadiomicsGLDM(image, mask, **settings)
        gldmFeatures.enableAllFeatures()

        # print('Will calculate the following GLDM features: ')
        # for f in gldmFeatures.enabledFeatures.keys():
        #   print('  ', f)
        #   print(getattr(gldmFeatures, 'get%sFeatureValue' % f).__doc__)

        # print('Calculating GLDM features...')
        results = gldmFeatures.execute()
        # print('done')

        print('Calculated GLDM features: ')
        for (key, val) in six.iteritems(results):
            GLDMFeatureName = '%s_%s' % ('GLDM', key)
            if GLDMFeatureName not in feature_vectors[case_id]:
                feature_vectors[case_id][GLDMFeatureName] = val
            else:
                print('Error: GLDM key existing! %s'%GLDMFeatureName)
                # break
            # print('  ', key, ':', val)


        #
        # Show FirstOrder features, calculated on a LoG filtered image
        #
        if applyLog:
          sigmaValues = np.arange(5., 0., -.5)[::1]
          for logImage, imageTypeName, inputKwargs in imageoperations.getLoGImage(image, mask, sigma=sigmaValues):
            logFirstorderFeatures = firstorder.RadiomicsFirstOrder(logImage, mask, **inputKwargs)
            logFirstorderFeatures.enableAllFeatures()
            results = logFirstorderFeatures.execute()
            for (key, val) in np.iteritems(results):
                laplacianFeatureName = '%s_%s' % (imageTypeName, key)
                if laplacianFeatureName not in feature_vectors[case_id]:
                    feature_vectors[case_id][laplacianFeatureName] = val
                else:
                    print('Error: LoG key existing! %s'%laplacianFeatureName)
                    # break
                # print('  ', laplacianFeatureName, ':', val)
        #
        # Show FirstOrder features, calculated on a wavelet filtered image
        #
        if applyWavelet:
          for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image, mask):
            waveletFirstOrderFeaturs = firstorder.RadiomicsFirstOrder(decompositionImage, mask, **inputKwargs)
            waveletFirstOrderFeaturs.enableAllFeatures()
            results = waveletFirstOrderFeaturs.execute()
            print('Calculated firstorder features with wavelet ', decompositionName)
            for (key, val) in six.iteritems(results):
                waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                if waveletFeatureName not in feature_vectors[case_id]:
                    feature_vectors[case_id][waveletFeatureName] = val
                else:
                    print('Error: wavelet key existing! %s'%waveletFeatureName)
                    # break
                # print('  ', waveletFeatureName, ':', val)

        mask = None
        image = None
    return feature_vectors
# check the required packages to be installed
# check_requirements()

# using dicom2nrrd_convertor to convert dicom to nrrd
# patients_dicom_path = "/mnt/iDriveShare/hossein/2_patientData/IMAGES"

def extractor(image_array, mask_array, applyLog = False,applyWavelet = False ):
    feature_vectors = {}
    cache_file = os.path.join(DATA_CACHE_PATH_OUT,'cache_%s.nrrd'%random.random())
    nrrd.write(cache_file, image_array)
    image = sitk.ReadImage(cache_file)
    nrrd.write(cache_file, mask_array)
    mask = sitk.ReadImage(cache_file)


    settings = {'binWidth': 25,
            'interpolator': None, #sitk.sitkBSpline,
            'resampledPixelSpacing': None}
    # interpolator = settings.get('interpolator')
    # resampledPixelSpacing = settings.get('resampledPixelSpacing')
    # if interpolator is not None and resampledPixelSpacing is not None:
    #     image, mask = imageoperations.resampleImage(image, mask, **settings)
    # bb, correctedMask = imageoperations.checkMask(image, mask)
    # if correctedMask is not None:
    #   mask = correctedMask
    # image, mask = imageoperations.cropToTumorMask(image, mask, bb)
    
    features = firstorder.RadiomicsFirstOrder(image, mask, **settings)
    # features.enableFeatureByName('Mean', True)
    features.enableAllFeatures()

    # print('Will calculate the following first order features: ')
    # for f in features.enabledFeatures.keys():
    #   print('  ', f)
    #   print(getattr(features, 'get%sFeatureValue' % f).__doc__)

    # print('Calculating first order features...')
    results = features.execute()
    # print('done')

    # print('Calculated first order features: ')
    for (key, val) in six.iteritems(results):
        firstOrderFeatureName = '%s_%s' % ('firstOrder', key)
        if firstOrderFeatureName not in feature_vectors:
            feature_vectors[firstOrderFeatureName] = val
        else:
            print('Error: firstOrder key existing! %s'%firstOrderFeatureName)
            # break
        # print('  ', key, ':', val)
    print('ADDED %s first Order Features'%len(results))
    #
    # Show Shape features
    #

    features = shape.RadiomicsShape(image, mask, **settings)
    features.enableAllFeatures()

    # print('Will calculate the following Shape features: ')
    # for f in features.enabledFeatures.keys():
    #   print('  ', f)
    #   print(getattr(features, 'get%sFeatureValue' % f).__doc__)

    # print('Calculating Shape features...')
    results = features.execute()
    # print('done')

    # print('Calculated Shape features: ')
    for (key, val) in six.iteritems(results):
        ShapeFeatureName = '%s_%s' % ('Shape', key)
        if ShapeFeatureName not in feature_vectors:
            feature_vectors[ShapeFeatureName] = val
        else:
            print('Error: shape key existing! %s'%ShapeFeatureName)
            # break
        # print('  ', key, ':', val)
    print('ADDED %s shape Features'%len(results))
    #
    # Show GLCM features: Gray Level Co-occurrence Matrix (GLCM) Features
    #
    features = glcm.RadiomicsGLCM(image, mask, **settings)
    features.enableAllFeatures()

    # print('Will calculate the following GLCM features: ')
    # for f in features.enabledFeatures.keys():
    #   print('  ', f)
    #   print(getattr(features, 'get%sFeatureValue' % f).__doc__)

    # print('Calculating GLCM features...')
    results = features.execute()
    # print('done')

    # print('Calculated GLCM features: ')
    for (key, val) in six.iteritems(results):
        GLCMFeatureName = '%s_%s' % ('GLCM', key)
        if GLCMFeatureName not in feature_vectors:
            feature_vectors[GLCMFeatureName] = val
        else:
            print('Error: GLCM key existing! %s'%GLCMFeatureName)
            # break
        # print('  ', key, ':', val)


    print('ADDED %s GLCM Features'%len(results))
    #
    # Show GLSZM features; Gray Level Size Zone Matrix (GLSZM) Features
    #
    features = glszm.RadiomicsGLSZM(image, mask, **settings)
    features.enableAllFeatures()

    # print('Will calculate the following GLSZM features: ')
    # for f in features.enabledFeatures.keys():
    #   print('  ', f)
    #   print(getattr(features, 'get%sFeatureValue' % f).__doc__)

    # print('Calculating GLSZM features...')
    results = features.execute()

    # print('Calculated GLSZM features: ')
    for (key, val) in six.iteritems(results):
        GLSZMFeatureName = '%s_%s' % ('GLSZM', key)
        if GLSZMFeatureName not in feature_vectors:
            feature_vectors[GLSZMFeatureName] = val
        else:
            print('Error: GLSZM key existing! %s'%GLSZMFeatureName)
            # break
        # print('  ', key, ':', val)
    print('ADDED %s GLSZ Features'%len(results))

    #
    # Show GLRLM features; Gray Level Run Length Matrix (GLRLM) Features
    #
    features = glrlm.RadiomicsGLRLM(image, mask, **settings)
    features.enableAllFeatures()

    # print('Will calculate the following GLRLM features: ')
    # for f in features.enabledFeatures.keys():
    #   print('  ', f)
    #   print(getattr(features, 'get%sFeatureValue' % f).__doc__)

    # print('Calculating GLRLM features...')
    results = features.execute()
    # print('done')

    # print('Calculated GLRLM features: ')
    for (key, val) in six.iteritems(results):
        GLRLMFeatureName = '%s_%s' % ('GLRLM', key)
        if GLRLMFeatureName not in feature_vectors:
            feature_vectors[GLRLMFeatureName] = val
        else:
            print('Error: GLRLM key existing! %s'%GLRLMFeatureName)
            # break
        # print('  ', key, ':', val)
    print('ADDED %s GLRM Features'%len(results))
    #
    # Show NGTDM features; Neighbouring Gray Tone Difference Matrix (NGTDM) Features
    #
    features = ngtdm.RadiomicsNGTDM(image, mask, **settings)
    features.enableAllFeatures()

    # print('Will calculate the following NGTDM features: ')
    # for f in features.enabledFeatures.keys():
    #   print('  ', f)
    #   print(getattr(features, 'get%sFeatureValue' % f).__doc__)

    # print('Calculating NGTDM features...')
    results = features.execute()
    # print('done')

    # print('Calculated NGTDM features: ')
    for (key, val) in six.iteritems(results):
        NGTDMFeatureName = '%s_%s' % ('NGTDM', key)
        if NGTDMFeatureName not in feature_vectors:
            feature_vectors[NGTDMFeatureName] = val
        else:
            print('Error: NGTDM key existing! %s'%NGTDMFeatureName)
            # break
        # print('  ', key, ':', val)
    print('ADDED %s NGTDM Features'%len(results))
    #
    # Show GLDM features; Gray Level Dependence Matrix (GLDM) Features
    #
    features = gldm.RadiomicsGLDM(image, mask, **settings)
    features.enableAllFeatures()

    # print('Will calculate the following GLDM features: ')
    # for f in features.enabledFeatures.keys():
    #   print('  ', f)
    #   print(getattr(features, 'get%sFeatureValue' % f).__doc__)

    # print('Calculating GLDM features...')
    results = features.execute()
    # print('done')

    # print('Calculated GLDM features: ')
    for (key, val) in six.iteritems(results):
        GLDMFeatureName = '%s_%s' % ('GLDM', key)
        if GLDMFeatureName not in feature_vectors:
            feature_vectors[GLDMFeatureName] = val
        else:
            print('Error: GLDM key existing! %s'%GLDMFeatureName)
            # break
        # print('  ', key, ':', val)

    print('ADDED %s GLDM Features'%len(results))

    #
    # Show FirstOrder features, calculated on a LoG filtered image
    #
    if applyLog:
      sigmaValues = np.arange(5., 0., -.5)[::1]
      cnt = 0
      for logImage, imageTypeName, inputKwargs in imageoperations.getLoGImage(image, mask, sigma=sigmaValues):
        cnt += 1
        features = firstorder.RadiomicsFirstOrder(logImage, mask, **inputKwargs)
        features.enableAllFeatures()
        results = features.execute()
        print('ADDEDING %s/%s Features...'%(cnt, len(logImage)))
        for (key, val) in six.iteritems(results):
            laplacianFeatureName = '%s_%s' % (imageTypeName, key)
            if laplacianFeatureName not in feature_vectors:
                feature_vectors[laplacianFeatureName] = val
            else:
                print('Error: LoG key existing! %s'%laplacianFeatureName)
                # break
            # print('  ', laplacianFeatureName, ':', val)
    #
    # Show FirstOrder features, calculated on a wavelet filtered image
    #
    if applyWavelet:
        cnt = 0
        for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image, mask):
            cnt+=1
            features = firstorder.RadiomicsFirstOrder(decompositionImage, mask, **inputKwargs)
            features.enableAllFeatures()
            results = features.execute()
            print('ADDEDING %s/%s WAVELET Features...'%(cnt, str(decompositionName)))
            print('Calculated firstorder features with wavelet ', decompositionName)
            for (key, val) in six.iteritems(results):
                waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                if waveletFeatureName not in feature_vectors:
                    feature_vectors[waveletFeatureName] = val
                else:
                    print('Error: wavelet key existing! %s'%waveletFeatureName)
                    # break
                # print('  ', waveletFeatureName, ':', val)

    mask = None
    image = None
    features = None
    os.remove(cache_file)
    print('DONE!')
    return feature_vectors
# check the required packages to be installed
# check_requirements()

# using dicom2nrrd_convertor to convert dicom to nrrd
# patients_dicom_path = "/mnt/iDriveShare/hossein/2_patientData/IMAGES"
    
def main():
    patients_nrrd_path = "./nrrds"
    # case_IDs = os.listdir(patients_dicom_path)
    # case_IDs = ['01']
    # print(case_IDs)
    # for case_ID in case_IDs:
    #     # dicom2nrrd_convertor.convert(patients_dicom_path, patients_nrrd_path, case_ID)
    #     pass 
    # # PyRadiomics Read Data from nrrd
    # # Instantiating the extractor, enable Image Types and Execute Features

    valid_IDs = [f for f in os.listdir(patients_nrrd_path) 
            if ('image.nrrd' in os.listdir(os.path.join(patients_nrrd_path,f))
               and 'mask.nrrd' in os.listdir(os.path.join(patients_nrrd_path,f)))]
    # valid_IDs = ['01']
    # print('valid_IDs: ',valid_IDs)

    feature_vectors = features_extractor(patients_nrrd_path, valid_IDs, applyLog = False,applyWavelet = False)
    feature_value_array = []
    header = feature_vectors[feature_vectors.keys()[0]].keys()
    feature_value_array . append(['caseID']+header)
    print (len(header))
    for key in feature_vectors:
        feature_value_array.append([key]+feature_vectors[key].values())
        if feature_vectors[key].keys() != header:
            print ('ERROR: Keys mis-match')
            print('########################################')
            print('########################################')
            print('########################################')
            print('##########                    ##########')
            print('##########      E R R O R     ##########')
            print('##########                    ##########')
            print('########################################')
            print('########################################')
            print('########################################')
    print('DONE!')

    with open('radiomics_feature_vectors.csv','w') as f:
        writer = csv.writer(f)
        writer.writerows(feature_value_array)
    # print(feature_vectors)
    # print(feature_value_array)
    # with open('radiomics_feature_vectors.json', 'w') as f:
    #     # json.dump(feature_vectors, f, ensure_ascii=False, indent=4)
    #     json.dump(feature_vectors, f)

if __name__ == '__main__':
    main()
'''
Enable/Disable all or selected image types 
Refer To Image Types Below.
- Original: 
    No filter applied
- Wavelet: 
    Wavelet filtering, yields 8 decompositions per level (all possible combinations of applying either
    a High or a Low pass filter in each of the three dimensions.
    See also :py:func:`~radiomics.imageoperations.getWaveletImage`
- LoG: 
    Laplacian of Gaussian filter, edge enhancement filter. Emphasizes areas of gray level change, where sigma
    defines how coarse the emphasised texture should be. A low sigma emphasis on fine textures (change over a
    short distance), where a high sigma value emphasises coarse textures (gray level change over a large distance).
    See also :py:func:`~radiomics.imageoperations.getLoGImage`
- Square: 
    Takes the square of the image intensities and linearly scales them back to the original range.
    Negative values in the original image will be made negative again after application of filter.
- SquareRoot: 
    Takes the square root of the absolute image intensities and scales them back to original range.
    Negative values in the original image will be made negative again after application of filter.
- Logarithm: 
    Takes the logarithm of the absolute intensity + 1. Values are scaled to original range and
    negative original values are made negative again after application of filter.
- Exponential: 
    Takes the the exponential, where filtered intensity is e^(absolute intensity). Values are
    scaled to original range and negative original values are made negative again after application of filter.
- Gradient: 
    Returns the gradient magnitude.
- LBP2D: 
    Calculates and returns a local binary pattern applied in 2D.
- LBP3D: 
    Calculates and returns local binary pattern maps applied in 3D using spherical harmonics. Last returned
    image is the corresponding kurtosis map.
'''
