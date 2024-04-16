
import sys
import nibabel as nib
import numpy as np

sys.path.insert(0, '/Users/matty_gee/Dropbox/Projects/toolbox/toolbox')
from images import get_nifti_info, save_as_nifti
from mvpa import run_sl, sl_mds

#-------------------------------------------------------------------------------------------
# get brain data, masks
#-------------------------------------------------------------------------------------------

mask_dir = '/Volumes/synapse/projects/SocialSpace/Projects/SNT-fmri_place/Masks'

# subject data & mask files
sub_dir = '/Volumes/synapse/projects/SocialSpace/Projects/SNT-fmri_place/Samples/Initial/GLMs_mask-thr50/GLMs/01'
sub_func_fname = f'{sub_dir}/beta_4d_resampled.nii'
sub_mask_fname = f'{sub_dir}/mask_resampled.nii'

# hpc mask
hpc_mask_fname = f'{mask_dir}/Repl_bilateral_HPC_harvardoxford_maxprob-thr0-1mm.nii'


#-------------------------------------------------------------------------------------------
# run searchlight
#-------------------------------------------------------------------------------------------

# define kernel
sl_kernel = sl_mds

# define broadcast variables
bcvar = {'n_components': 2, 'metric': True}

# run main searchlight function
sl_result = run_sl(sub_func_fname, sub_mask_fname, sl_kernel, bcvar=bcvar, \
                   other_masks=hpc_mask_fname, \
                   shape='ball', radius=3, \
                   min_prop=0.10, num_sls=10)

# save image
dims, vox_size, affine_matrix = get_nifti_info(sub_func_fname)
save_as_nifti(sl_result, 'test.nii', affine_matrix, vox_size)



