import glob
from os.path import exists
import numpy   as np 
import pandas  as pd
import scipy   as sp
import nibabel as nib
import nilearn
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker, NiftiMapsMasker
from nilearn.image import load_img, get_data, new_img_like, math_img
from nilearn.masking import compute_brain_mask
import matplotlib.pyplot as plt


#---------------------------------------------------------------------------------------------------------
# nifti io
#---------------------------------------------------------------------------------------------------------


def load_nifti(nifti_fname):  
    return nib.load(nifti_fname)

def get_nifti_info(nifti):
    ''' return dimensions, voxel size and affine matrix of a nifti '''
    if isinstance(nifti, str): nifti = nib.load(nifti)
    dims = nifti.get_fdata().shape
    vox_size = nifti.header.get_zooms()[:3] # just get xyz
    affine_matrix = nifti.affine
    return dims, vox_size, affine_matrix 

def make_nifti(brain_data, affine_matrix, vox_size):
    '''
        Make a nifti from matrix

        Arguments
        ---------
        brain_data : matrix
            Brain data to make nifti from
        affine_matrix : matrix
            Affine matrix

        Returns
        -------
        nifti 

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''

    # compatibility
    brain_data = brain_data.astype('double')
    brain_data[np.isnan(brain_data)] = 0

    # create the volume image
    brain_nii = nib.Nifti1Image(brain_data, affine_matrix)  
    hdr = brain_nii.header 
    if brain_data.ndim == 4: hdr.set_zooms((vox_size[0], vox_size[1], vox_size[2], 0))
    else:                    hdr.set_zooms((vox_size[0], vox_size[1], vox_size[2]))
    
    return brain_nii

def make_nifti_compatible_crossplatform(nifti_fname, overwrite=False):
    '''
        Make a nifti compatible with cross-platform (nilearn & spm)

        Arguments
        ---------
        nifti_fname : str
            Path to nifti file
        out_file : str
            Path to save nifti file

        Returns
        -------
        nothing 

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''

    # read in the nifti file
    brain_nii = nib.load(nifti_fname)
    brain_data = brain_nii.get_fdata()
    affine_matrix = brain_nii.affine
    vox_size = brain_nii.header.get_zooms()

    # make the nifti again
    brain_nii = make_nifti(brain_data, affine_matrix, vox_size)
    if overwrite: 
        out_fname = nifti_fname
    else: 
        out_fname = f"{nifti_fname.split('.')[0]}_matlab.nii" 
    nib.save(brain_nii, out_fname) # save

def save_as_nifti(brain_data, output_name, affine_mat, vox_size):
    '''
        Save a brain matrix as a nifti file

        Arguments
        ---------
        brain_data : numpy array
            An array of brain data to be converted into a nifti
        output_name : str
            Name to save the image to
        affine_mat : numpy array
            Affine matrix
        vox_size : tuple
            Voxel size in 3 dimensions

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    brain_data = brain_data.astype('double')  # Convert the output into a precision format that can be used by other applications
    brain_data[np.isnan(brain_data)] = 0  # Exchange nans with zero to ensure compatibility with other applications
    brain_nii = nib.Nifti1Image(brain_data, affine_mat)  # create the volume image
    hdr = brain_nii.header  # get a handle of the .nii file's header
    if brain_data.ndim == 4: hdr.set_zooms((vox_size[0], vox_size[1], vox_size[2], 0))
    else:                    hdr.set_zooms((vox_size[0], vox_size[1], vox_size[2]))
    nib.save(brain_nii, output_name)  # Save the volume  


#---------------------------------------------------------------------------------------------------------
# masking
#---------------------------------------------------------------------------------------------------------


def compute_wholebrain_mask(nifti, threshold=0.10):
    # wrapper for nilearn masking function to compute whole-brain mask
    return nilearn.masking.compute_brain_mask(nifti, 
                                              threshold=threshold, 
                                              connected=False, 
                                              opening=2, 
                                              mask_type='whole-brain')

def compute_gm_mask(nifti, threshold=0.25):
    # wrapper for nilearn masking function to compute grey matter mask
    return nilearn.masking.compute_brain_mask(nifti, 
                                              threshold=threshold, 
                                              connected=False, 
                                              opening=2, 
                                              mask_type='gm')

def masks_union(mask_list):
    return nil.masking.intersect_masks(mask_list, threshold=0, connected=False)

def masks_intersection(mask_list):
    return nilearn.masking.intersect_masks(mask_list, threshold=1, connected=False)

def get_incl_gm_mask(func_img, gm_thresh=0.25):
    ''' returns a gm mask x included voxel mask for func image '''
    # incl any voxel != 0: these are voxels had some computation done to them
    incl_mask = new_img_like(func_img, (get_data(func_img) != 0) * 1)
    # get a gm mask with specific threshold
    gm_mask = compute_brain_mask(func_img, mask_type='gm', threshold=gm_thresh, connected=False)
    # intersect the gm & voxel inclusion masks
    incl_gm_mask = nilearn.masking.intersect_masks([incl_mask, gm_mask], threshold=1, connected=False)
    return incl_gm_mask

def get_masked_img(func_img, mask_img):
    '''
        Returns an image but with only masked voxels

        Arguments
        ---------
        func_img : str or nifti
            Image to mask
        mask_img : str or nifti
            Image to mask with

        Returns
        -------
        nifti 
            Masked image

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    if isinstance(func_img, str): func_img = nib.load(func_img)
    if isinstance(mask_img, str): mask_img = nib.load(mask_img)
    
    # make a masker object (force a resample)
    masker = NiftiMasker(mask_img=mask_img, 
                         target_affine=func_img.affine, 
                         target_shape=func_img.get_fdata().shape) 
    
    # get masked volume
    masked_data = masker.fit_transform(func_img) # 2D

    # transform back to 4D
    func_img_masked = masker.inverse_transform(masked_data)

    # if input was 3D, return 3D
    if (func_img.ndim == 3) & (func_img_masked.ndim == 4):
        func_img_masked = new_img_like(func_img, get_data(func_img_masked)[:,:,:,0], func_img.affine)

    return func_img_masked

def get_voxels_from_mask(func_img, mask_img, resample_to_func=False, standardize=False):
    '''
        mask_img: 3d nii (ideally already resampled to correct dims)
        sub_img: 4d nii
        returns: array of shape (time_points, voxels)
    '''
    if resample_to_func:
        sub_dims, _, sub_affine = get_nifti_info(func_img)
        masker = NiftiMasker(mask_img=mask_img,
                             target_affine=sub_affine, target_shape=sub_dims[:3],
                             standardize=standardize)
    else:
        masker = NiftiMasker(mask_img=mask_img, standardize=standardize)
    return masker.fit_transform(func_img)

def find_roi_labels_ixs(labels, roi_name='Hippocampus'):
    return [np.where(np.array(labels) == label)[0][0] for label in [l for l in labels if roi_name in l]]

def get_timeseries(func_img, 
                   mask=None,
                   mask_type='gm-template',
                   radius=None,
                   target_shape=None,
                   target_affine=None,
                   smoothing_fwhm=None,
                   tr=None,
                   detrend=False, 
                   standardize=False, 
                   low_pass=None, 
                   high_pass=None,
                   confounds=None, 
                   standardize_confounds=False, 
                   high_variance_confounds=False,
                   memory_level=2,
                   verbose=0):
    '''
        Extracts timeseries from functional image w/ nilearn masking functions
        Puts the different use cases into standard function
        Only parameters I am likely to use are specified below

        Arguments
        ---------
        func_img : nifti
            Image to mask
        mask : image or coordinates (optional, default=None)
            If passed, use in appropriate masker instance 
            NOTE: if 'mask' is an image and diff resolution from 'func_img', 'func_img' resampled to 'mask' unless target_shape and/or target_affine are provided
        mask_type : str (optional, default='gm-template')
            'sphere' : coordinates for spherical mask
            'roi' : mask image
            'map' : overlapping volumes
            'atlas' : overlapping volumes
        radius : _type_ (optional, default=None)
            Radius of sphere 
        smoothing_fwhm : float (optional, default=None)
            Smoothing kernel 
        standardize : bool (optional, default=False)
            _description_ 
        confounds : _type_ (optional, default=None)
            _description_ 

        Returns
        -------
        timeseries : array 
            Timeseries, with shape=(num_regions, num_timepoints)
        masker : nilearn object
            the masker used

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    if mask_type=='roi':
    
        masker = NiftiMasker(mask_img=mask,
                             smoothing_fwhm=smoothing_fwhm, t_r=tr,
                             detrend=detrend, standardize=standardize, 
                             low_pass=low_pass, high_pass=high_pass,
                             standardize_confounds=standardize_confounds,
                             high_variance_confounds=high_variance_confounds,
                             memory='nilearn_cache', memory_level=memory_level, verbose=verbose)   
        
    elif mask_type in ['whole-brain', 'gm']:

        masker = NiftiMasker(mask_strategy=mask_type + '-template',
                             smoothing_fwhm=smoothing_fwhm, t_r=tr,
                             detrend=detrend, standardize=standardize, 
                             low_pass=low_pass, high_pass=high_pass,
                             standardize_confounds=standardize_confounds,
                             high_variance_confounds=high_variance_confounds,
                             memory='nilearn_cache', memory_level=memory_level, verbose=verbose)    
        
    elif mask_type=='sphere':
        
        if radius is None: radius = 8 # default radius size
        masker = NiftiSpheresMasker(mask, radius=radius,
                                    smoothing_fwhm=smoothing_fwhm, t_r=tr,
                                    detrend=detrend, standardize=standardize, 
                                    low_pass=low_pass, high_pass=high_pass,
                                    standardize_confounds=standardize_confounds,
                                    high_variance_confounds=high_variance_confounds,
                                    memory='nilearn_cache', memory_level=memory_level, verbose=verbose)   
    
    elif mask_type=='map':
        
        # https://nilearn.github.io/dev/auto_examples/03_connectivity/plot_signal_extraction.html
        masker = NiftiMapsMasker(maps_img=mask, resampling_target="data",
                                 mask_img=None,
                                 smoothing_fwhm=smoothing_fwhm, t_r=tr,
                                 detrend=detrend, standardize=standardize, 
                                 low_pass=low_pass, high_pass=high_pass,
                                 standardize_confounds=standardize_confounds,
                                 high_variance_confounds=high_variance_confounds,
                                 memory='nilearn_cache', memory_level=memory_level, verbose=verbose)         
    
    elif mask_type=='atlas':
        
        # https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_probabilistic_atlas_extraction.html#sphx-glr-auto-examples-03-connectivity-plot-probabilistic-atlas-extraction-py
        masker = NiftiLabelsMasker(labels_img=mask, resampling_target="data",
                                   mask_img=None,
                                   smoothing_fwhm=smoothing_fwhm, t_r=tr,
                                   detrend=detrend, standardize=standardize, 
                                   low_pass=low_pass, high_pass=high_pass,
                                   standardize_confounds=standardize_confounds,
                                   high_variance_confounds=high_variance_confounds,
                                   memory='nilearn_cache', memory_level=memory_level, verbose=verbose)
        
    timeseries = masker.fit_transform(func_img, confounds=confounds)

    return timeseries.T, masker #TODO standardize shape of outputs across functions...

def create_brain_mask(nii_fname, mask_type='whole-brain'):
    brain_mask = nilearn.masking.compute_brain_mask(nii_fname, threshold=0.1, 
                                                    connected=True, opening=2, 
                                                    mask_type=mask_type)
    _, vox_size, affine = get_nifti_info(nii_fname)
    print(f'{os.path.dirname(nii_fname)}/brain_mask.nii')
    save_as_nifti(brain_mask.get_fdata(), f'{os.path.dirname(nii_fname)}/brain_mask.nii', affine, vox_size)


#---------------------------------------------------------------------------------------------------------
# transformations
#---------------------------------------------------------------------------------------------------------


def resample_nifti(nifti_path, target_affine, target_shape, interpolation='nearest'):
    nii = nib.load(nifti_path)
    resampled_nii = nil.image.resample_img(nii, target_affine=target_affine, 
                                           target_shape=target_shape, interpolation=interpolation)    
    return resampled_nii    


#---------------------------------------------------------------------------------------------------------
# quality control
#---------------------------------------------------------------------------------------------------------


def output_mask_jpg(mask_path):

    # get mask
    mask_img  = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # get info
    dims      = mask_data.shape
    vox_size  = mask_img.header.get_zooms()[0:3]
    n_voxels  = np.sum(mask_data == 1)

    # plot & save
    mask_name = mask_path.split('/')[-1].split('.')[0]
    fig, ax = plt.subplots(figsize=(10, 7))
    mask_fig = nil.plotting.plot_roi(mask_img, title=f'{mask_name}\n- dims={dims}\n- vox_size={vox_size}\n- n_voxels={n_voxels}', 
                                        annotate=False, draw_cross=False, axes=ax, 
                                        output_file=f'{mask_name}.jpg')
    
def calc_roi_snr(func_fname, rois, nonbrain_coords=None, nonbrain_radius=10):
    '''
        Get temporal and spatial signal-to-noise ratios for rois in a functional image

        Arguments
        ---------
        func_fname : str
            Path to functional image
        rois : list of str  
            Paths to ROIs to get TSNR & SSNR for
        nonbrain_coords : tuple (optional)
            Where to put nonbrain ROI for SSNR
            If None, use a plausible nonbrain location
        nonbrain_radius : int (optional)
            Radius of nonbrain ROI
            Default: 10

        Returns
        -------
        pd.DataFrame 
            Contains TSNR and SSNR for each ROI, with shape: (num_rois, 3)

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''

    # define an roi outside of the brain
    if nonbrain_coords is None:
        func_shape = get_nifti_info(func_fname)[0][:3]
        nonbrain_coords = (func_shape[0]-5, func_shape[1]-5, func_shape[2]-5)
    nonbrain_masker     = NiftiSpheresMasker([nonbrain_coords], radius=nonbrain_radius)
    nonbrain_timeseries = nonbrain_masker.fit_transform(func_fname)

    # loop over defined rois
    snr_df = pd.DataFrame(columns=['roi', 'ssnr', 'tsnr'])
    for r, roi in enumerate(rois):
        brain_timeseries = get_voxels_from_mask(roi, func_fname)

        # spatial snr
        ssnr = np.nanmean(brain_timeseries) / np.nanstd(nonbrain_timeseries)

        # temporal snr
        voxelwise_tsnr = np.nanmean(brain_timeseries, 0) / np.nanstd(brain_timeseries, 0)
        tsnr = np.nanmean(voxelwise_tsnr)

        snr_df.loc[r, :] = [roi, ssnr, tsnr]

    return snr_df