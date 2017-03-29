# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as snd
import seaborn as sns

from skimage import img_as_float, morphology, measure
from skimage.morphology import disk, opening
from skimage.morphology import opening
from skimage.filters import sobel
from skimage.feature import peak_local_max
from skimage.filters.rank import enhance_contrast_percentile
from skimage.color import rgb2hsv
from skimage.morphology import reconstruction
from skimage.exposure import rescale_intensity
from skimage.measure import label
from astropy.table import Table
from scipy import spatial


def mac_det(fi, mp_threshold, patch_threshold, percb, percc, tcr, ds,px, plot=True, morph=False, testing=True):
    """
    Function for detecting macropores (cracks & biopores) to analyse their spatial arrangement & 
    matrix interaction
    
    Line 133:162 are adapted from the preprocessor of the echoRD-model by C. Jackisch. 
    For further informations see: https://github.com/cojacoo/echoRD_model/tree/master/echoRD
    file "macropore_ini.py"
    
    Parameters
    ----------
    fi : input image ( in .png-format, either as rgb or rgba image)
    mp_threshold : lower limit[0] for removing small patches, 
        upper limit [1] for seperating cracks from other objects
    patch_threshold : min [0] and max [1] of the desired patch size limits
    percb : value up to which percentile grey values among to biopores 
        (0.125 shows good results for brighter soil matrix)
    percc : lower [0] & upper [1] percentile for crack contrast enhancement
    tcr : value up to which percentile grey values among to cracks
    ds : disk size for morphological opening
    px : actual length of one pixel in input image [mm]
    plot : True/False: whether results should be plotted (default: True)
    morph : if True the morphology of detected biopores will be plotted, otherwise pores are displayed as 
            scatterplot and distinguished whether stained or not (default: False)
    testing = if True no distances are calculated and only the detected macropores are plotted to reduce 
        computing time threshold adjustment (default), otherwise all distances are computed
    
    Output
    ------
    Dictionary with following keys:
    'biopores' : labeled biopores
    'biopores_centroidxy' : x/y-coordinates of detected biopores
    'biopores_stained_centroidxy' : x/y-coordinates of detected stained biopores
    'biopores_area' : area of detected biopores (number of pixels)
    'biopores_diameter' : diameter of detected biopores (diameter of circle with same area [mm])
    'cracks' : labeled cracks
    'distance_matrix_biopore' : distance of each image pixel to nearest biopore [mm]
    'distance_matrix_stained_biopore' : distance of each image to nearest stained biopore [mm]
    'biopore_matrix_interaction' : distance of pixels from stained patches including at least one 
               biopore to nearest stained biopore [mm] (estimation of biopore-matrix interaction)
    'stained_patches' : labeled blue-stained patches
    'patches_with_biopores' : detected blue-stained patches including at least one biopore
    'table' : summary table with number and main propertiesd of detected biopores
    'stained_index' : index of stained biopores
    'unstained_index' : index of unstained biopores
    
    """
    
    im_raw = snd.imread(fi)   # load image   
    sim = np.shape(im_raw)
    if sim[2]==4:
        imrgb=im_raw[:,:,:3]
    else:
        imrgb=im_raw
    imhsv = rgb2hsv(imrgb)   # convert RGB image to HSV color-space
    img = imhsv[:,:,2]       # extract value channel
    im = img_as_float(imrgb) # load image as float for detection of stained patches
    sim = np.shape(im)       # extract dimensions of input image
    
    
    # biopore detection
    # morphological reconstruction for detecting holes inside the picture (according to general example 
    # "filling holes and detecting peaks" from scikit-image)
    seed = np.copy(img)
    seed[1:-1, 1:-1] = img.max()
    mask = img

    filled = reconstruction(seed, mask, method='erosion')
    holes=img-filled
    
    # rescale and extract biopores
    holes_resc=rescale_intensity(holes,out_range=(0.0,1))

    thresh=np.percentile(holes_resc,percb)
    holes_resc[holes_resc>thresh]=1 
    holes_resc[holes_resc<thresh]=0

    bp_lab=label(holes_resc,neighbors=8, background=1)
    bp_lab[bp_lab==-1]=0

    # remove objects smaller than threshold
    bp_lab_clean = morphology.remove_small_objects(bp_lab, min_size=mp_threshold[0])
    
    
    # crack detection
    # enhance contrast for dark spots
    cr_ecp=enhance_contrast_percentile(img, disk(3), p0=percc[0], p1=percc[1])
    threshcr=np.percentile(cr_ecp,tcr)
    cr_ecp[cr_ecp>threshcr]=255 
    cr_ecp[cr_ecp<threshcr]=1

    # morphological opening for bridging small gaps and remove small white pixels
    cr_ecpo=opening(cr_ecp, disk(ds))
    cr_lab=label(cr_ecpo,neighbors=8, background=255)
    cr_lab[cr_lab==-1]=0

    # remove objects smaller than threshold
    cr_label = morphology.remove_small_objects(cr_lab, min_size=mp_threshold[1])
    
    # remove biopores inside labeled cracks
    bp_label=bp_lab_clean-cr_label.dot(100)
    bp_label[bp_label<0]=0
    
    
    # detect and label blue stained patches
    # calculate difference of channels to extract blue stained patches
    dim=abs(im[:,:,1]-im[:,:,0])
    
    # discard low contrasts
    dim[dim<0.2]=0.0

    # filter to local maxima for further segmentation
    # process segmentation according to sobel function of skimage
    image_max = snd.maximum_filter(dim, size=5, mode='constant')

    elevation_map = sobel(dim)

    markers = np.zeros_like(dim)
    markers[image_max < 0.1] = 2
    markers[image_max > 0.2] = 1

    segmentation = morphology.watershed(elevation_map, markers)

    segmentation = snd.binary_fill_holes(1-(segmentation-1))

    # clean patches below theshold
    patches_cleaned = morphology.remove_small_objects(segmentation, patch_threshold[0])
    labeled_patches = label(patches_cleaned)
    sizes = np.bincount(labeled_patches.ravel())[1:] #first entry (background) discarded

    # reanalyse for large patches and break them by means of watershed segmentation
    idx=np.where(sizes>patch_threshold[1])[0]+1
    labeled_patches_large=labeled_patches*0
    idy=np.in1d(labeled_patches,idx).reshape(np.shape(labeled_patches))
    labeled_patches_large[idy]=labeled_patches[idy]
    distance = snd.distance_transform_edt(labeled_patches_large)
    footp=int(np.round(np.sqrt(patch_threshold[1])/100)*100)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((footp, footp)),labels=labeled_patches_large)
    markers = snd.label(local_maxi)[0]
    labels_broken_large = morphology.watershed(-distance, markers, mask=labeled_patches_large)
    labeled_patches[idy]=labels_broken_large[idy]+np.max(labeled_patches)
    
    
    # measure regionproperties of biopores
    meas_bp=measure.regionprops(bp_label, intensity_image=None)

    bp_labels = np.unique(bp_label)[1:]
    
    bp_centroidx = bp_labels.astype(np.float64)
    bp_centroidy = bp_labels.astype(np.float64)
    bp_area = bp_labels.astype(np.float64)
    bp_diameter = bp_labels.astype(np.float64)
    
    # extract regionprops for each labeled biopore
    for i in np.arange(len(bp_labels)):
        bp_centroidx[i], bp_centroidy[i]=meas_bp[i]['centroid']
        bp_area[i]=meas_bp[i]['area']*px
        bp_diameter[i]=meas_bp[i]['equivalent_diameter']*px
        
    bp_centroidxy = np.stack((bp_centroidx,bp_centroidy), axis=-1)
    
    
    # extract biopores inside stained areas = "stained biopores"
    stain_info=np.zeros(len(bp_centroidxy))
    rbp_centroidxy=np.around(bp_centroidxy).astype(int)

    for i in np.arange(len(bp_centroidxy)):
        if labeled_patches[rbp_centroidxy[i,0],rbp_centroidxy[i,1]]>0:
            stain_info[i]=1
        else:
            stain_info[i]=2
    
    stained=np.where(stain_info==1)
    unstained=np.where(stain_info==2)
    
    # select value of stained patches including an biopore
    bp_stained=np.around(bp_centroidxy[stained]).astype(int)
    label_value=np.zeros(len(bp_stained)).astype(int)

    for i in np.arange(len(bp_stained)):
        label_value[i]=labeled_patches[bp_stained[i,0], bp_stained[i,1]]
    
    # remove labeled patches without any biopore
    label_withbp=np.copy(labeled_patches)

    for i in np.arange(len(label_value)):
        label_withbp[label_withbp==label_value[i]]=-1

    label_withbp[label_withbp!=-1]=0
    label_withbp[label_withbp==-1]=1
    
    
    # distance calculations
    if testing==False:
        # Compute Euclidian distance for each pixel to nearest biopore
        m_bp_dist = np.zeros((sim[0],sim[1]))

        for i in np.arange(sim[0]):
            for j in np.arange(sim[1]):
                matrixp1=[i,j]
                m_bp_dist[i,j]=spatial.KDTree(bp_centroidxy).query(matrixp1,p=2)[0]
    
        # compute Euclidian distance for each pixel to nearest stained biopore
        m_stbp_dist=np.zeros((sim[0],sim[1]))

        for i in np.arange(sim[0]):
            for j in np.arange(sim[1]):
                matrixp1=[i,j]
                m_stbp_dist[i,j]=spatial.KDTree(bp_centroidxy[stained]).query(matrixp1,p=2)[0]
    
        # compute Euclidian distance to nearest stained biopore for each pixel of stained areas including a biopore ~ biopore-matrix interaction
        stp_stbp_dist = np.zeros((sim[0],sim[1]))

        for i in np.arange(sim[0]):
            for j in np.arange(sim[1]):
                if label_withbp[i,j]!=0:
                    matrixp3=[i,j]
                    stp_stbp_dist[i,j,]=spatial.KDTree(bp_centroidxy[stained]).query(matrixp3,p=2)[0]
            
                else:
                    stp_stbp_dist[i,j]=np.nan
    
    # table for output
    sbp_diameter=bp_diameter[stained]
            
    t1='All biopores','Stained biopores','Cracks'
    t2=len(bp_diameter),len(sbp_diameter), len(np.unique(cr_label)[1:])
    t3 = len(bp_diameter[bp_diameter<2]),len(sbp_diameter[sbp_diameter<2]),np.nan
    t4 = len(bp_diameter[bp_diameter>=6]),len(sbp_diameter[sbp_diameter>=6]),np.nan
    t5 =len(bp_diameter[bp_diameter>=2]),len(sbp_diameter[sbp_diameter>=2]),np.nan

    attr=[t1,t2,t3,np.subtract(t5,t4),t4]
    mc_t=Table(attr,names=('Properties','Sum','<2mm','2-6mm','>6mm'),meta=None)
    
    
    # plot results
    if plot==True:
        #colors for plot
        from matplotlib.colors import ListedColormap
        ghostwhite=(248/255,248/255,255/255)
        blue=(31/255,119/255,180/255)
        cmap=ListedColormap([ghostwhite, blue])

        if testing==False:  
            # flatten arrays for kernel density estimate plot
            m_bp_distall = np.ravel(m_bp_dist*px)
            m_stbp_distall = np.ravel(m_stbp_dist*px)
            stp_stbp_distall = np.ravel(stp_stbp_dist*px)

            #plot
            sns.set_style('white')
            plt.figure(figsize=(15,3))
            ax1=plt.subplot(141)
            plt.imshow(imrgb)
            plt.axis('off')
            plt.title('Input image')
        
            plt.subplot(142, sharex=ax1, sharey=ax1)
            plt.imshow(labeled_patches, vmin=0, vmax=1, cmap=cmap)
            plt.imshow(imrgb, alpha=0.5)
            if morph==True:
                plt.imshow(bp_label, vmin=0, vmax=1,cmap='binary', alpha=0.5)
            else:
                plt.scatter(bp_centroidxy[unstained][:,1],bp_centroidxy[unstained][:,0] ,color='black', s=10,label='unstained')
                plt.scatter(bp_centroidxy[stained][:,1], bp_centroidxy[stained][:,0] ,color='red', s=15, label='stained')
                plt.legend(bbox_to_anchor=[1,0], ncol=2)
            plt.axis('off')
            plt.title('Labeled patches & Biopores')
     
            plt.subplot(143, sharex=ax1, sharey=ax1)
            plt.imshow(cr_label,vmin=0, vmax=1,cmap='binary')
            plt.imshow(labeled_patches, vmin=0, vmax=1,cmap=cmap, alpha=0.5)
            plt.imshow(imrgb, alpha=0.2)
            plt.axis('off')
            plt.title('Labeled patches & Cracks')
        
            plt.subplot(144)
            sns.kdeplot(m_bp_distall, cut=0, label='All pores')
            if len(stained[0])>0:
                sns.kdeplot(m_stbp_distall, cut=0, label='Stained pores' ,alpha=0.5)
                sns.kdeplot(stp_stbp_distall[~np.isnan(stp_stbp_distall)], cut=0, label='Biopore-matrix interaction' ,alpha=0.5)
            plt.title('Frequency distribution of calculated distances')
        
            plt.show()
        
            print(mc_t)
        else:
            #plot
            sns.set_style('white')
            plt.figure(figsize=(15,4))
            ax1=plt.subplot(131)
            plt.imshow(imrgb)
            plt.axis('off')
            plt.title('Input image')
        
            plt.subplot(132, sharex=ax1, sharey=ax1)
            plt.imshow(labeled_patches, vmin=0, vmax=1, cmap=cmap)
            plt.imshow(imrgb, alpha=0.5)
            if morph==True:
                plt.imshow(bp_label, vmin=0, vmax=1,cmap='binary', alpha=0.5)
            else:
                plt.scatter(bp_centroidxy[unstained][:,1],bp_centroidxy[unstained][:,0] ,color='black', s=10,label='unstained')
                plt.scatter(bp_centroidxy[stained][:,1], bp_centroidxy[stained][:,0] ,color='red', s=15, label='stained')
                plt.legend(bbox_to_anchor=[1,0], ncol=2)
            plt.axis('off')
            plt.title('Labeled patches & Biopores')
     
            plt.subplot(133, sharex=ax1, sharey=ax1)
            plt.imshow(cr_label,vmin=0, vmax=1,cmap='binary')
            plt.imshow(labeled_patches, vmin=0, vmax=1,cmap=cmap, alpha=0.5)
            plt.imshow(imrgb, alpha=0.2)
            plt.axis('off')
            plt.title('Labeled patches & Cracks')
        
            plt.show()
        
            print(mc_t)
    
    
    # results for output
    mc_res={}
    mc_res['biopores'],mc_res['cracks'], mc_res['stained_patches'],mc_res['patches_with_biopores']=bp_label, cr_label, labeled_patches, label_withbp
    mc_res['biopores_centroidxy'], mc_res['biopores_stained_centroidxy']=bp_centroidxy, bp_centroidxy[stained]
    mc_res['biopores_area'], mc_res['biopores_diameter']=bp_area, bp_diameter
    if testing==False:
        mc_res['distance_matrix_biopore'], mc_res['distance_matrix_stained_biopore'], mc_res['biopore_matrix_interaction']=m_bp_dist*px, m_stbp_dist*px, stp_stbp_dist*px
    mc_res['table']=mc_t
    mc_res['stained_index'], mc_res['unstained_index']=stained, unstained
    
    return mc_res