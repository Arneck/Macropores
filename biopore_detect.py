# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as snd

from skimage import img_as_float
from skimage.color import rgb2hsv
from skimage.morphology import reconstruction
from skimage.exposure import rescale_intensity
from skimage import morphology
from skimage.measure import label
from skimage import measure
from astropy.table import Table
from scipy import spatial
import seaborn as sns
from scipy import ndimage
from skimage.filters import sobel
from skimage.feature import peak_local_max


def biop_det(fi, mp_threshold, patch_threshold, perc, plot):
   '''Function for detecting biopores to analyse their spatial arrangement & matrix interaction
      Line 64:101 are adapted from Conrad Jackisch. For further informations see: https://github.com/cojacoo/
      file "macropore_ini.py"'''
      
   # fi is input image
   # mp_threshold is lower limit for removing small patches
   # perc = values within percentile among to biopores (0.125 shows good results 
   #        for brighter soil matrix)
   # plot = True/False: whether results should be plotted
   # patch_threshold are min [0] and max [1] of the desired patch size limits
   
   imrgb = snd.imread(fi)                 
   imhsv = rgb2hsv(imrgb) # convert RGB image to HSV color-space
   img = imhsv[:,:,2] # extract value channel
   im = img_as_float(imrgb) 
   sim = np.shape(im)
   
   # morphological reconstruction for detecting holes inside the picture (according to general example 
   # "filling holes and detecting peaks" from scikit-image)
   seed = np.copy(img)
   seed[1:-1, 1:-1] = img.max()
   mask = img

   filled = reconstruction(seed, mask, method='erosion')
   holes=img-filled
   
   # rescale and extract macropores
   holes_resc=rescale_intensity(holes,out_range=(0.0,1))

   thresh=np.percentile(holes_resc,perc)
   holes_resc[holes_resc>thresh]=1 
   holes_resc[holes_resc<thresh]=0

   bp_label=label(holes_resc,neighbors=8, background=1)
   bp_label[bp_label==-1]=0

   # remove objects smaller than threshold
   bp_label_clean = morphology.remove_small_objects(bp_label, min_size=mp_threshold[0])
   
   
   # detect and label blue stained patches
   # calculate difference of channels to extract blue stained patches
   dim=abs(im[:,:,1]-im[:,:,0])
   
   # discard low contrasts
   dim[dim<0.2]=0.0

   # filter to local maxima for further segmentation
   # process segmentation according to sobel function of skimage
   # patch_threshold = 51 #define theshold for macropore identification
   image_max = ndimage.maximum_filter(dim, size=5, mode='constant')

   elevation_map = sobel(dim)

   markers = np.zeros_like(dim)
   markers[image_max < 0.1] = 2
   markers[image_max > 0.2] = 1

   segmentation = morphology.watershed(elevation_map, markers)

   segmentation = ndimage.binary_fill_holes(1-(segmentation-1))

   # clean patches below theshold
   patches_cleaned = morphology.remove_small_objects(segmentation, patch_threshold[0])
   labeled_patches, lab_num = ndimage.label(patches_cleaned)
   sizes = np.bincount(labeled_patches.ravel())[1:] #first entry (background) discarded

   # reanalyse for large patches and break them by means of watershed segmentation
   idx=np.where(sizes>patch_threshold[1])[0]+1
   labeled_patches_large=labeled_patches*0
   idy=np.in1d(labeled_patches,idx).reshape(np.shape(labeled_patches))
   labeled_patches_large[idy]=labeled_patches[idy]
   distance = ndimage.distance_transform_edt(labeled_patches_large)
   footp=int(np.round(np.sqrt(patch_threshold[1])/100)*100)
   local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((footp, footp)),labels=labeled_patches_large)
   markers = ndimage.label(local_maxi)[0]
   labels_broken_large = morphology.watershed(-distance, markers, mask=labeled_patches_large)
   labeled_patches[idy]=labels_broken_large[idy]+np.max(labeled_patches)
   
   
   # measure regionproperties of biopores
   meas_bp=measure.regionprops(bp_label_clean, intensity_image=None)

   bp_labels = np.unique(bp_label_clean)[1:]

   bp_centroidx = bp_labels.astype(np.float64)
   bp_centroidy = bp_labels.astype(np.float64)
   bp_area = bp_labels.astype(np.float64)
   bp_perimeter = bp_labels.astype(np.float64)
   bp_diameter = bp_labels.astype(np.float64)

   # extract regionprops for each labeled patch
   for i in np.arange(len(bp_labels)):
       bp_centroidx[i], bp_centroidy[i]=meas_bp[i]['centroid']
       bp_area[i]=meas_bp[i]['area']
       bp_perimeter[i]=meas_bp[i]['perimeter']
       bp_diameter[i]=meas_bp[i]['equivalent_diameter']
   
   # spatial analysis
   bp_mindist = bp_labels.astype(np.float64)
   bp_maxdist = bp_labels.astype(np.float64)
   bp_meandist = bp_labels.astype(np.float64)
   bp_mediandist = bp_labels.astype(np.float64)

   # calculate distances of centroids
   for i in np.arange(len(bp_labels)):
       bp_cxm=np.ma.array(bp_centroidx, mask=False)
       bp_cym=np.ma.array(bp_centroidy, mask=False)
       bp_cxm.mask[i]=True
       bp_cym.mask[i]=True
       bp_mindist[i]=np.sqrt(np.min((bp_cxm - bp_centroidx[i])**2 + (bp_cym - bp_centroidy[i])**2))
       bp_maxdist[i]=np.sqrt(np.max((bp_cxm - bp_centroidx[i])**2 + (bp_cym - bp_centroidy[i])**2))
       bp_meandist[i]=np.mean(np.sqrt((bp_cxm - bp_centroidx[i])**2 + (bp_cym - bp_centroidy[i])**2))
       bp_mediandist[i]=np.median(np.sqrt((bp_cxm - bp_centroidx[i])**2 + (bp_cym - bp_centroidy[i])**2))
   
   
   # Compute Euclidian distance for each pixel to nearest biopore
   bp_centroidxy = np.stack((bp_centroidx,bp_centroidy), axis=-1)
   m_bp_dist = np.zeros((sim[0],sim[1]))

   for i in np.arange(sim[0]):
       for j in np.arange(sim[1]):
           matrixp=[i,j]
           m_bp_dist[i,j]=spatial.KDTree(bp_centroidxy).query(matrixp,p=2)[0]
   
   # compute Euclidian distance for each pixel inside stained areas to nearest macropore
   stp_bp_dist=np.zeros((sim[0],sim[1]))

   for i in np.arange(sim[0]):
       for j in np.arange(sim[1]):
           if labeled_patches[i,j]!=0:
               stpp=[i,j]
               stp_bp_dist[i,j]=spatial.KDTree(bp_centroidxy).query(stpp,p=2)[0]
           
           else:
               stp_bp_dist[i,j]=np.nan
   
   # store results in List for output
   res_ano=['bp_res[0]=centroidx,bp_res[1]=centroidy, bp_res[2]=area, bp_res[3]=perimeter,bp_res[4]=diameter, bp[5]=pixel dist to biop, bp[6]=stainded patch dist to biop']
   bp_res=[res_ano, bp_centroidx, bp_centroidy, bp_area, bp_perimeter, bp_diameter, m_bp_dist, stp_bp_dist]
   
   
   # plot results
   if plot==True:
       
       # create Table for comparison
       t1=[len(bp_area[bp_area<2])]
       t3=len(bp_area[bp_area>=6])
       t2=len(bp_area[bp_area>=2])-t3
       t4=[round(np.min(bp_mindist))]
       t5=[round(np.max(bp_maxdist))]
       t6=[round(np.mean(bp_meandist))]
       t7=[round(np.median(bp_mediandist))]

       t2, t3=[t2],[t3]

       bp_t=Table([t1,t2,t3,t4,t5,t6,t7], names=('<2mm','2-6mm','>6mm','mindist','maxdist','meandist','mediandist'),meta=None)
       
       # concatenate arrays for plot
       stp_bp_distall = np.concatenate(stp_bp_dist)
       m_bp_distall = np.concatenate(m_bp_dist)

       # create frame for second plot
       frame=np.zeros((sim[0],sim[1]))
       end1,end2=sim[0]-3, sim[1]-3
       frame[:4,:]=5
       frame[end1:,:]=5
       frame[:,:3]=5
       frame[:,end2:]=5
       
       #plot
       plt.figure(figsize=(15,4))

       plt.subplot(131)
       plt.imshow(imrgb)
       plt.axis('off')
       plt.title('input image')

       plt.subplot(132)
       plt.imshow(bp_label_clean,vmin=0, vmax=1,cmap='binary')
       plt.imshow(labeled_patches, vmin=0, vmax=1,cmap='binary', alpha=0.2)
       plt.imshow(frame, alpha=0.1)
       plt.axis('off')
       plt.title('labeled patches & biopores')

       plt.subplot(133)
       sns.kdeplot(m_bp_distall, cut=0, label='all pixels')
       sns.kdeplot(stp_bp_distall, cut=0, label='stained pixels' ,alpha=0.5)
       sns.set_style("white")
       plt.title('distance to nearest macropore\nfrequency distribution')
       
       plt.show()
       
       print('biopore properties')
       print(bp_t)
       
   return bp_res

