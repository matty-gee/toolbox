#!/usr/bin/env python3


def get_snr(func_img, rois, nonbrain_coords=(55,55,55), nonbrain_radius=10):
    '''
        TODO: debug this...
        Get temporal and spatial signal-to-noise ratios for rois in a functional image

        Arguments
        ---------
        func_img : str
            Path to functional image
        rois : list of str  
            Paths to ROIs to get TSNR & SSNR for
            TODO: make this flexible to accept coordinates too... 
        nonbrain_coords : tuple (optional)
            Where to put nonbrain ROI for SSNR
            Default: (55,55,55)
        nonbrain_radius : int (optional)
            Radius of nonbrain ROI
            Default: 10

        Returns
        -------
        pd.DataFrame 
            Contains TSNR and SSNR for each ROI, with shape: (num_rois, 2)

        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''


    # get an roi outside of the brain
    nonbrain_masker     = NiftiSpheresMasker([nonbrain_coords], radius=nonbrain_radius)
    nonbrain_timeseries = nonbrain_masker.fit_transform(func_img)
    
    # loop over defined rois
    snr_df = pd.DataFrame(columns=['roi', 'ssnr', 'tsnr'])
    for r, roi in enumerate(rois):
        brain_timeseries = get_voxels_from_mask(roi, func_img)
        
        # spatial snr
        ssnr = np.nanmean(brain_timeseries) / np.nanstd(nonbrain_timeseries)

        # temporal snr
        voxelwise_tsnr = np.nanmean(brain_timeseries, 0) / np.nanstd(brain_timeseries, 0)
        tsnr = np.nanmean(voxelwise_tsnr)
        
        snr_df.loc[r, :] = [ssnr, tsnr]
        
    return snr_df

