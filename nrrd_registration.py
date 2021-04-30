#nrrd_registration
import os
import numpy
import SimpleITK as sitk
import matplotlib.pyplot as plt

def sitk_show(img, title=None, margin=0.0, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    #spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    #extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()

def sitk_convert():
  patients_dicom_path = "/var/www/devDocuments/hossein/Galenus/data/TSPINE_nrrd"
  p_1 = "0007680"
  p_2 = "0035549"
  path_p1 = "%s/%s/ct_image.nrrd"%(patients_dicom_path,p_1)
  path_p2 = "%s/%s/ct_image.nrrd"%(patients_dicom_path,p_2)
  print path_p1

  # Slice index to visualize with 'sitk_show'
  idxSlice = 45

  # int label to assign to the segmented gray matter
  labelGrayMatter = 1

  nrrd_p1 = sitk.ReadImage(path_p1)
  nrrd_p2 = sitk.ReadImage(path_p2)

  print nrrd_p1.GetSize(), nrrd_p2.GetSize()
  # sitk_show(sitk.Tile(nrrd_p1[:, :, idxSlice],
  #                          nrrd_p2[:, :, idxSlice],
  #                          (2, 1, 0)))
  sitk_show(sitk.Tile(nrrd_p1[:, 255, :]))
  sitk_show(sitk.Tile(nrrd_p2[:, 255, :]))
  smooth_p1 = sitk.CurvatureFlow(image1=nrrd_p1,
                                        timeStep=0.125,
                                        numberOfIterations=5)
  # print nrrd_p1.shape
  smooth_p2 = sitk.CurvatureFlow(image1=nrrd_p2,
                                        timeStep=0.125,
                                        numberOfIterations=5)

  # sitk_show(sitk.Tile(smooth_p1[:, :, idxSlice], 
                           # smooth_p2[:, :, idxSlice], 
                           # (2, 1, 0)))

  lstSeeds = [(165, 178, idxSlice),
              (98, 165, idxSlice),
              (205, 125, idxSlice),
              (173, 205, idxSlice)]

  imgSeeds = sitk.Image(smooth_p2)

  for s in lstSeeds:
      imgSeeds[s] = 10000

  # sitk_show(imgSeeds[:, :, idxSlice])


