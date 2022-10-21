import glob
from os.path import exists
import numpy   as np 
import pandas  as pd
import scipy   as sp
import nibabel as nib
import nilearn as nil
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker, NiftiMapsMasker
from nilearn.image import load_img, get_data, new_img_like, math_img
from nilearn.masking import compute_brain_mask

#---------------------------------------------------------------------------------------------------------
# io
#---------------------------------------------------------------------------------------------------------

def load_nifti(nifti_fname):  
    return nib.load(nifti_fname)


def get_nifti_info(nifti):
    ''' return dimensions, voxel size and affine matrix of a nifti '''
    if isinstance(nifti, str): nifti = nib.load(nifti)
    dims = nifti.get_data().shape
    vox_size = nifti.header.get_zooms()[0:3] # just get xyz
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
    brain_data = brain_data.astype('double')  # Convert the output into a precision format that can be used by other applications
    brain_data[np.isnan(brain_data)] = 0  # Exchange nans with zero to ensure compatibility with other applications
    brain_nii = nib.Nifti1Image(brain_data, affine_matrix)  # create the volume image
    hdr = brain_nii.header  # get a handle of the .nii file's header
    if brain_data.ndim == 4: hdr.set_zooms((vox_size[0], vox_size[1], vox_size[2], 0))
    else:                    hdr.set_zooms((vox_size[0], vox_size[1], vox_size[2]))
    return brain_nii


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

def get_incl_gm_mask(func_img, gm_thresh=0.25):
    ''' returns a gm mask x included voxel mask for func image '''
    # incl any voxel != 0: these are voxels had some computation done to them
    incl_mask = new_img_like(func_img, (get_data(func_img) != 0) * 1)
    # get a gm mask with specific threshold
    gm_mask = compute_brain_mask(func_img, mask_type='gm', threshold=gm_thresh, connected=False)
    # intersect the gm & voxel inclusion masks
    incl_gm_mask = nil.masking.intersect_masks([incl_mask, gm_mask], threshold=1, connected=False)
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
    if isinstance(mask_img, str): func_img = nib.load(mask_img)

    # a masker to extract (force a resample)
    masker = nil.input_data.NiftiMasker(mask_img=mask_img, 
                                        target_affine=func_img.affine, 
                                        target_shape=func_img.get_fdata().shape) 
    # get masked volume (fit_transform extracts to 2d)
    masked_data = masker.fit_transform(func_img) 
    # get back to brain volume (inverse_transform outputs in 4d)
    func_img_masked = masker.inverse_transform(masked_data)
    # if input was in 3d, output in 3d
    if func_img.ndim == 3:
        data = get_data(func_img_masked)[:,:,:,0] 
        func_img_masked = new_img_like(func_img, data, func_img.affine)
    return func_img_masked


def get_voxels_from_mask(mask_img, sub_img, resample_to_sub=False, standardize=False):
    '''
        mask_img: 3d nii (ideally already resampled to correct dims)
        sub_img: 4d nii
        returns: array of shape (time_points, voxels)
    '''
    if resample_to_sub:
        sub_dims, _, sub_affine = get_nifti_info(sub_img)
        masker = NiftiMasker(mask_img=mask_img, 
                             target_affine=sub_affine, target_shape=sub_dims[0:3],
                             standardize=standardize)
    else:
        masker = NiftiMasker(mask_img=mask_img, standardize=standardize)
    return masker.fit_transform(sub_img)


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
                   low_pass=None, high_pass=None,
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
            'sphere' :
            'roi' : 
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

    return timeseries.T, masker

#---------------------------------------------------------------------------------------------------------
# operations
#---------------------------------------------------------------------------------------------------------

def resample_nifti(nifti_path, target_affine, target_shape, interpolation='nearest'):

    nii = nib.load(nifti_path)
    resampled_nii = nil.image.resample_img(nii, target_affine=target_affine, target_shape=target_shape, interpolation=interpolation)    
    return resampled_nii    