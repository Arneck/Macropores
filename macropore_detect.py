# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as snd
import seaborn as sns

from skimage import img_as_float
from skimage.color import rgb2hsv
from skimage.morphology import reconstruction
from skimage.exposure import rescale_intensity
from skimage import morphology
from skimage.measure import label
from skimage import measure
from astropy.table import Table
from scipy import spatial
from scipy import ndimage
from skimage.filters import sobel
from skimage.feature import peak_local_max
from skimage.filters.rank import enhance_contrast_percentile
from skimage.morphology import disk
from skimage.morphology import opening
from skimage.morphology import skeletonize


def mac_det(fi, mp_threshold, patch_threshold, perc, percc, tcr, ds, plot):
    """
    Function for detecting macropores (cracks&biopores) to analyse their spatial arrangement & matrix interaction
    
    Line 96:132 are adapted from Conrad Jackisch. For further informations see: https://github.com/cojacoo/
    file "macropore_ini.py"
    
    parameters
    ----------
    fi = input image
    mp_threshold = lower limit[0] for removing small patches, upper limit [1] for seperating cracks from other objects
    percp = value up to which percentile grey values among to biopores (0.125 shows good results for brighter soil matrix)
    percc = lower [0] & upper [1] percentile for crack contrast enhancement
    plot = True/False: whether results should be plotted
    patch_threshold = min [0] and max [1] of the desired patch size limits
    tcr = value up to which percentile grey values among to cracks
    ds = disk size for morphological opening
    """
    
    imrgb = snd.imread(fi)                 
    imhsv = rgb2hsv(imrgb)   # convert RGB image to HSV color-space
    img = imhsv[:,:,2]       # extract value channel
    im = img_as_float(imrgb) 
    sim = np.shape(im)
    
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

    thresh=np.percentile(holes_resc,perc)
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
    meas_bp=measure.regionprops(bp_label, intensity_image=None)

    bp_labels = np.unique(bp_label)[1:]
    
    bp_centroidx = bp_labels.astype(np.float64)
    bp_centroidy = bp_labels.astype(np.float64)
    bp_area = bp_labels.astype(np.float64)
    bp_diameter = bp_labels.astype(np.float64)
    
    # extract regionprops for each labeled biopore
    for i in np.arange(len(bp_labels)):
        bp_centroidx[i], bp_centroidy[i]=meas_bp[i]['centroid']
        bp_area[i]=meas_bp[i]['area']
        bp_diameter[i]=meas_bp[i]['equivalent_diameter']
        
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
    
    
    # spatial analysis of all biopores
    bp_mindist = bp_labels.astype(np.float64)
    bp_maxdist = bp_labels.astype(np.float64)
    bp_meandist = bp_labels.astype(np.float64)
    bp_mediandist = bp_labels.astype(np.float64)

    # calculate distances of all centroids
    for i in np.arange(len(bp_labels)):
        bp_cxm=np.ma.array(bp_centroidx, mask=False)
        bp_cym=np.ma.array(bp_centroidy, mask=False)
        bp_cxm.mask[i]=True
        bp_cym.mask[i]=True
        bp_mindist[i]=np.sqrt(np.min((bp_cxm - bp_centroidx[i])**2 + (bp_cym - bp_centroidy[i])**2))
        bp_maxdist[i]=np.sqrt(np.max((bp_cxm - bp_centroidx[i])**2 + (bp_cym - bp_centroidy[i])**2))
        bp_meandist[i]=np.mean(np.sqrt((bp_cxm - bp_centroidx[i])**2 + (bp_cym - bp_centroidy[i])**2))
        bp_mediandist[i]=np.median(np.sqrt((bp_cxm - bp_centroidx[i])**2 + (bp_cym - bp_centroidy[i])**2))
    
    
    # spatial analysis of stained biopores
    sbp_mindist = bp_labels[stained].astype(np.float64)
    sbp_maxdist = bp_labels[stained].astype(np.float64)
    sbp_meandist = bp_labels[stained].astype(np.float64)
    sbp_mediandist = bp_labels[stained].astype(np.float64)
    sbp_centroidx, sbp_centroidy = bp_centroidx[stained], bp_centroidy[stained]
    
    # calculate distances of  stained centroids
    for i in np.arange(len(sbp_mindist)):
        sbp_cxm=np.ma.array(sbp_centroidx, mask=False)
        sbp_cym=np.ma.array(sbp_centroidy, mask=False)
        sbp_cxm.mask[i]=True
        sbp_cym.mask[i]=True
        sbp_mindist[i]=np.sqrt(np.min((sbp_cxm - sbp_centroidx[i])**2 + (sbp_cym - sbp_centroidy[i])**2))
        sbp_maxdist[i]=np.sqrt(np.max((sbp_cxm - sbp_centroidx[i])**2 + (sbp_cym - sbp_centroidy[i])**2))
        sbp_meandist[i]=np.mean(np.sqrt((sbp_cxm - sbp_centroidx[i])**2 + (sbp_cym - sbp_centroidy[i])**2))
        sbp_mediandist[i]=np.median(np.sqrt((sbp_cxm - sbp_centroidx[i])**2 + (sbp_cym - sbp_centroidy[i])**2))
    
    
    # skeletonize cracks for distance measurements
    cr_01=cr_label.copy()
    cr_01[cr_01!=0]=1
    cr_skel=skeletonize(cr_01)
    
    # extract xy values of skeletonized cracks + corresponding value of labeled crack
    sim=np.shape(img)
    crxy=np.zeros((1,3))

    for i in np.arange(sim[0]):
        for j in np.arange(sim[1]):
            if cr_skel[i,j]==True:
                value=cr_label[i,j]
                crxy=np.append(crxy,[[i,j,value]],axis=0)

    crxy=np.delete(crxy,crxy[0],axis=0)
    
    # Compute Euclidean distances of cracks
    cr_skel= np.zeros((len(crxy)))
    cry=crxy[:,0]
    crx=crxy[:,1]
    cr_value=crxy[:,2]

    cr_mindist=cr_skel.astype(np.float64)
    cr_maxdist=cr_skel.astype(np.float64)
    cr_meandist=cr_skel.astype(np.float64)
    cr_mediandist=cr_skel.astype(np.float64)


    # calculate distances of skeleton points
    for i in np.arange(len(cr_skel)):
        crxm=np.ma.array(crx, mask=False)
        crym=np.ma.array(cry, mask=False)
        value_i=cr_value[i]
        crxm.mask[np.where(cr_value==value_i)]=True
        crym.mask[np.where(cr_value==value_i)]=True
        
        cr_mindist[i]=np.sqrt(np.min((crxm - crx[i])**2 + (crym - cry[i])**2))
        cr_maxdist[i]=np.sqrt(np.max((crxm - crx[i])**2 + (crym - cry[i])**2))
        cr_meandist[i]=np.mean(np.sqrt((crxm - crx[i])**2 + (crym - cry[i])**2))
        cr_mediandist[i]=np.median(np.sqrt((crxm - crx[i])**2 + (crym - cry[i])**2))
    
    
    # compute Euclidean distance for each pixel to nearest biopore
    m_bp_dist = np.zeros((sim[0],sim[1]))

    for i in np.arange(sim[0]):
        for j in np.arange(sim[1]):
            matrixp=[i,j]
            m_bp_dist[i,j]=spatial.KDTree(bp_centroidxy).query(matrixp,p=2)[0]
    
    # compute Euclidean distance for each pixel inside stained areas to nearest macropore
    stp_bp_dist=np.zeros((sim[0],sim[1]))

    for i in np.arange(sim[0]):
        for j in np.arange(sim[1]):
            if labeled_patches[i,j]!=0:
                stpp=[i,j]
                stp_bp_dist[i,j]=spatial.KDTree(bp_centroidxy).query(stpp,p=2)[0]
            
            else:
                stp_bp_dist[i,j]=np.nan
    
    # compute Euclidean distance to nearest stained biopore for each pixel of stained areas including a biopore
    pwbp_stbp_dist = np.zeros((sim[0],sim[1]))

    for i in np.arange(sim[0]):
        for j in np.arange(sim[1]):
            if label_withbp[i,j]!=0:
                stpp2=[i,j]
                pwbp_stbp_dist[i,j,]=spatial.KDTree(bp_centroidxy[stained]).query(stpp2,p=2)[0]
            
            else:
                pwbp_stbp_dist[i,j]=np.nan
    
    # table for output
    sbp_diameter=bp_diameter[stained]
    
    if len(stained[0])>1:
        t6_2 =round(np.nanmin(sbp_mindist))
        t7_2 =round(np.nanmax(sbp_maxdist))
        t8_2 =round(np.nanmean(sbp_meandist))
        t9_2 =round(np.nanmedian(sbp_mediandist))
    else:
        t6_2=np.nan
        t7_2=np.nan
        t8_2=np.nan
        t9_2=np.nan
    
    if len(np.unique(cr_label)[1:])>1:
        t6_3 =round(np.nanmin(cr_mindist))
        t7_3 =round(np.nanmax(cr_maxdist))
        t8_3 =round(np.nanmean(cr_meandist))
        t9_3 =round(np.nanmedian(cr_mediandist))
    else:
        t6_3=np.nan
        t7_3=np.nan
        t8_3=np.nan
        t9_3=np.nan
        
    t1='all biopores','stained biopores','cracks'
    t2=len(bp_diameter),len(sbp_diameter), len(np.unique(cr_label)[1:])
    t3 = len(bp_diameter[bp_diameter<2]),len(sbp_diameter[sbp_diameter<2]),np.nan
    t4 = len(bp_diameter[bp_diameter>=6]),len(sbp_diameter[sbp_diameter>=6]),np.nan
    t5 =len(bp_diameter[bp_diameter>=2]),len(sbp_diameter[sbp_diameter>=2]),np.nan
    t6 = round(np.nanmin(bp_mindist)),t6_2,t6_3
    t7 = round(np.nanmax(bp_maxdist)),t7_2,t7_3
    t8 = round(np.nanmean(bp_meandist)),t8_2,t8_3
    t9 = round(np.nanmedian(bp_mediandist)),t9_2, t9_3

    attr=[t1,t2,t3,np.subtract(t5,t4),t4,t6,t7,t8,t9]
    mc_t=Table(attr,names=('properties','sum','<2mm','2-6mm','>6mm','mindist','maxdist','meandist','mediandist'),meta=None)
    
    
    # plot results
    if plot==True:
        
        # concatenate arrays for plot
        stp_bp_distall = np.concatenate(stp_bp_dist)
        m_bp_distall = np.concatenate(m_bp_dist)
        pwbp_stbp_distall = np.concatenate(pwbp_stbp_dist)
        
        # create frame for second plot
        frame=np.zeros((sim[0],sim[1]))
        end1,end2=sim[0]-3, sim[1]-3
        frame[:4,:]=5
        frame[end1:,:]=5
        frame[:,:3]=5
        frame[:,end2:]=5
        
        #plot
        plt.figure(figsize=(15,3))
        
        plt.subplot(141)
        plt.imshow(imrgb)
        plt.axis('off')
        plt.title('input image')
        
        plt.subplot(142)
        plt.imshow(bp_label,vmin=0, vmax=1,cmap='binary')
        plt.imshow(labeled_patches, vmin=0, vmax=1,cmap='binary', alpha=0.2)
        plt.imshow(frame, alpha=0.1)
        plt.axis('off')
        plt.title('labeled patches & biopores')
        
        plt.subplot(143)
        plt.imshow(cr_label,vmin=0, vmax=1,cmap='binary')
        plt.imshow(labeled_patches, vmin=0, vmax=1,cmap='binary', alpha=0.2)
        plt.imshow(frame, alpha=0.1)
        plt.axis('off')
        plt.title('labeled patches & cracks')
        
        plt.subplot(144)
        sns.set_style("white")
        sns.kdeplot(m_bp_distall, cut=0, label='all pixels')
        if len(stained[0])>0:
            sns.kdeplot(stp_bp_distall, cut=0, label='stained pixels' ,alpha=0.5)
        sns.kdeplot(pwbp_stbp_distall, cut=0, label='stained pixels with biopore-\nstained biopore' ,alpha=0.5)
        plt.title('distance to nearest biopore\nfrequency distribution')
        
        plt.show()
        
        print(mc_t)
    
    # results for output
    mc_res={}
    mc_res['biopores'],mc_res['cracks'], mc_res['stained_patches'],mc_res['patches_with_bp']=bp_label, cr_label, labeled_patches, label_withbp
    mc_res['centroidxy']=bp_centroidxy
    mc_res['area'], mc_res['diameter']=bp_area, bp_diameter
    mc_res['stained_index'], mc_res['unstained_index']=stained, unstained
    mc_res['matrix_bp'], mc_res['stained_bp'], mc_res['stained_bpstained']=m_bp_dist, stp_bp_dist, pwbp_stbp_dist
    mc_res['bp_mindist'], mc_res['bp_maxdist'],mc_res['bp_meandist'], mc_res['bp_mediandist']=bp_mindist, bp_maxdist,bp_meandist, bp_mediandist
    mc_res['cr_mindist'], mc_res['cr_maxdist'],mc_res['cr_meandist'], mc_res['cr_mediandist']=cr_mindist, cr_maxdist,cr_meandist, cr_mediandist
    mc_res['sbp_mindist'], mc_res['sbp_maxdist'],mc_res['sbp_meandist'], mc_res['sbp_mediandist']=sbp_mindist, sbp_maxdist,sbp_meandist, sbp_mediandist
    mc_res['table']=mc_t
    
    return mc_res