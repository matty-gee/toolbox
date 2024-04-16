import time
import numpy as np
import pandas as pd
import nilearn as nil
from nilearn import datasets
from nilearn.maskers import NiftiSpheresMasker, NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker
from nilearn.connectome import ConnectivityMeasure
import networkx as nx 

# my modules
from images import get_timeseries, get_nifti_info, save_as_nifti
from general_utils import load_pickle, save_json, pickle_file

#-------------------------------------------------------------------------------------------
# timeseries
#-------------------------------------------------------------------------------------------


def save_roi_timeseries(sub_id, lsa_dir):
    
    # define masks
    mask_dir   = '/sc/arion/projects/k23/Masks' # minerva
    masks_dict = {'HO_Cortl': [datasets.fetch_atlas_harvard_oxford('cortl-maxprob-thr25-2mm'), 'atlas'], # 96 regions
                  'HO_Sub':   [datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm'), 'atlas'], # 21 regions
                  'Juelich':  [datasets.fetch_atlas_juelich('maxprob-thr25-2mm', symmetric_split=True), 'atlas'], # 131 regions
                  'Shen': [load_pickle(f'{mask_dir}/Shen_1mm_368_parcellation.pkl'), 'atlas'], # 368 regions
                  'Apriori_L-IFG_radius8': [[(-45, 41, -2)], 'sphere', 8], 
                  'Apriori_R-IFG_radius8': [[(51, 50, 1)], 'sphere', 8],
                  'Apriori_L-HPC_Tavares_radius8': [[(-22, -15, -18)], 'sphere', 8],
                  'Apriori_R-Precuneus_Tavares_radius8': [[(-14, 60, 26)], 'sphere', 8],
                  'Apriori_L-HPC_avg_radius8': [[(-23, -26, -11)], 'sphere', 8],
                  'Apriori_L-HPC_ant': [f'{mask_dir}/L-HPC_ant_harvardoxford_maxprob-thr25-1mm.nii', 'roi'],
                  'Apriori_L-HPC_post': [f'{mask_dir}/L-HPC_post_harvardoxford_maxprob-thr25-1mm.nii', 'roi'],
                  'Apriori_L-HPC_mid': [f'{mask_dir}/L-HPC_mid_harvardoxford_maxprob-thr25-1mm.nii', 'roi'],
                  'Apriori_R-HPC_ant': [f'{mask_dir}/R-HPC_ant_harvardoxford_maxprob-thr25-1mm.nii', 'roi'],
                  'Apriori_R-HPC_post': [f'{mask_dir}/R-HPC_post_harvardoxford_maxprob-thr25-1mm.nii', 'roi'],
                  'Apriori_R-HPC_mid': [f'{mask_dir}/R-HPC_mid_harvardoxford_maxprob-thr25-1mm.nii', 'roi']}

    beta_img  = f'{lsa_dir}/subs/sub-P{sub_id}/beta_4d.nii'
    out_fname = f'{lsa_dir}/roi_timeseries/{sub_id}_roi_timeseries.xlsx'
    
    # extract timeseries
    timeseries_df = pd.DataFrame()
    roi_labels    = []
    for mask_name, mask_info in masks_dict.items():

        mask      = mask_info[0]
        mask_type = mask_info[1]

        # multiple regions at once
        # -- an atlas has fully separated regions
        # -- a map has overlapping regions [dbl check how these are separated]
        if mask_type in ['atlas', 'map']: 
            n_labels = len(np.unique(mask.maps.get_fdata()))
            print(f'{mask_type} {mask_name}: extracting timeseries; num labels={n_labels}') 
            timeseries, _ = get_timeseries(beta_img, mask=mask.maps, mask_type=mask_type)            
            labels = np.array(mask_info[0].labels)
            assert n_labels != len(labels), f'WARNING: label mismatch - {n_labels} != {len(labels)}'
            labels = [mask_name + '_' + l for l in labels[labels != 'Background']]
            
        # single region
        elif mask_type in ['sphere', 'roi']:
            print(f'{mask_type} {mask_name}: extracting timeseries; num labels=1')
            if mask_type == 'sphere':
                timeseries, _ = get_timeseries(beta_img, mask=mask, mask_type='sphere', radius=mask_info[2])
            elif mask_type == 'roi':
                timeseries, _ = get_timeseries(beta_img, mask=mask, mask_type='roi')
                timeseries    = np.mean(timeseries, 0).reshape(1,-1) # average across voxels
            labels = [mask_name]

        # print & organize
        print(f'{mask_type} {mask_name}: extracted timeseries w/ shape {timeseries.shape}; num labels={len(labels)}')
        roi_labels.extend([l.translate(str.maketrans('', '', ''.join([' ', ',', '(', ')', "'", '-', '/']))) for l in labels]) 
        timeseries_df = pd.concat([timeseries_df, pd.DataFrame(timeseries)], ignore_index=True)

        # was getting errors, where it the same atlas' timeseries would be used multiple times
        # -- slowing it down a little helped...?
        time.sleep(30) 

    # output
    timeseries_df.columns= [f'trial_{t:02d}' for t in range(1,64)]
    timeseries_df.insert(0, 'roi', roi_labels)
    timeseries_df.to_excel(out_fname, index=False) # output timeseries


#-------------------------------------------------------------------------------------------
# correlations between timeseries
#-------------------------------------------------------------------------------------------


def compute_roi_correlation(ts_fname, kind='correlation'):
    '''
        kind: correlation or partial_correlation
    '''

    # clean up
    kind = kind.replace('_', ' ') # partial_correlation -> partial correlation
    df   = pd.read_excel(ts_fname)
    rois = df['roi'].values
    ts   = df.iloc[:, 1:].values.T

    # compute correlations
    fc   = ConnectivityMeasure(kind=kind, vectorize=False, discard_diagonal=False).fit_transform([ts])[0]
    fc_z = np.arctanh(fc) # z-transform

    # output
    fc_z = pd.DataFrame(fc_z, columns=rois, index=rois)
    fc_fname = ts_fname.replace('timeseries/', 'fc/') # folder name
    fc_fname = fc_fname.replace('timeseries', kind.replace(' ', '_') + '_z') # file name
    fc_z.to_excel(fc_fname)


def compute_region_to_voxel_fc(img_fname, mask, radius=8):
    '''
        Parameters
        __________
        img_fname : str
        mask : str if roi_type=='roi'; 3 item tuple if roi_type=='sphere'
        radius : 
    '''

    # get region and whole brain timeseries with shape (voxels, trials)
    brain_timeseries, brain_masker = get_timeseries(img_fname, mask_type='whole-brain')

    # check if mask is a string
    if isinstance(mask, str):
        region_timeseries, _ = get_timeseries(img_fname, mask=mask, mask_type='roi')
        region_timeseries    = np.mean(region_timeseries, 0).reshape(1,-1) # average across voxels
    else:
        region_timeseries, _ = get_timeseries(img_fname, mask=mask, mask_type='sphere', radius=radius)
    print(f'Region timeseries shape: {region_timeseries.shape}')
    print(f'Brain timeseries shape: {brain_timeseries.shape}')
    assert region_timeseries.shape[1] == brain_timeseries.shape[1], 'Incorrect shape of the timeseries'
    
    # perform the seed-to-voxel correlation
    corrs   = (np.dot(brain_timeseries, region_timeseries.T) / region_timeseries.T.shape[0]) # dot((voxels, trials), (trials, voxels))
    corrs_z = np.arctanh(corrs) # fisher z-transform
    fc_img  = brain_masker.inverse_transform(corrs_z.T) # returns 4d
    fc_img  = nil.image.index_img(fc_img, 0)
    print(f'FC image shape: {fc_img.shape}')
    
    # # output image
    # if roi_type == 'sphere':
    #     out_str = f'{mask[0][0]}_{mask[0][1]}_{mask[0][2]}_radius{radius}'
    # elif roi_type == 'roi':
    #     out_str = mask_name
    # fc_img.to_filename(f'{out_dir}/{sub_id}_{out_str}_correlation_z.nii.gz')

    return fc_img


#-------------------------------------------------------------------------------------------
# network analysis
#-------------------------------------------------------------------------------------------


class GraphProperties:

    '''
        By KK & MS
        takes into a functional connectivity matrix and computes its different graph properties 
    '''

    def __init__(self, matrix, node_names, node_attributes=None, graph_attributes=None):

        '''
            upon initialization find graph properties

            parameters
            ----------
            matrix : 
            node_names : 
            node_attributes : 
            graph_attributes : 

        '''
        # # Convert upper matrix to 2D matrix if upper matrix given
        # if matrix.ndim == 1:
        #     matrix = ut_vec_to_symm_mat(matrix)

        # generate graph and relabel nodes
        self.graph    = nx.from_numpy_matrix(matrix)
        index_mapping = dict(zip(np.arange(len(node_names)).astype(int), node_names))
        self.graph    = nx.relabel.relabel_nodes(self.graph, index_mapping)
        self.matrix   = matrix

        # Save graph/node/edge attributes
        if node_attributes is None: node_attributes = ['degree_centrality', 'betweenness_centrality', 
                                                       'closeness_centrality', 'eigenvector_centrality',
                                                       'clustering']
        for node_attr in node_attributes:
            print(f'Computing node attribute: {node_attr}')
            nx.set_node_attributes(self.graph, getattr(nx, node_attr)(self.graph), node_attr)            
                
        self.graph.attributes = {}
        if graph_attributes is None: graph_attributes = ['local_efficiency', 'global_efficiency']
                                                        # 'sigma', 'degree_assortativity_coefficient', 
                                                        #  'rich_club_coefficient']
        for graph_attr in graph_attributes:
            print(f'Computing graph attribute: {graph_attr}')
            if 'efficiency' in graph_attr:
                val = getattr(nx.algorithms.efficiency_measures, graph_attr)(self.graph)
            else:
                val = getattr(nx, graph_attr)(self.graph)
            self.graph.attributes[graph_attr] = val
        
    def find_communities(self, levels=1, assign=False):
        '''
            apply community detection algorithm

            parameters
            ----------
            levels : 
            assign : 
        '''
        comm_generator = nx.algorithms.community.girvan_newman(self.graph)
        for _ in np.arange(levels):
            comms = next(comm_generator)
        if assign:
            self.communities = comms
            for counter, comm in enumerate(comms, start=1):
                for node_name in comm:
                    nx.set_node_attributes(self.graph, {node_name: counter}, "community")
        return comms


def compute_graph_properties(fc_fname, atlas='HO', thresh=0.95, weighted=False):

    '''
        computes graph properties of functional connectivity
        
        parameters
        ----------
        fc_fname :
        atlas : 
        thresh : 
        weighted : 
    '''
    
    # input
    df = pd.read_excel(fc_fname, index_col=0)
    rois = df.columns.values
    atlas_rois = [l for l in rois if  f'{atlas}_' in l]
    
    # compute graph properties
    f = df.loc[atlas_rois, atlas_rois].values # extract only the atlas-specific fc values
    a = (f > np.percentile(f, thresh)) # adjacency matrix thresholded with a percentile...
    if weighted: a = a * f 
    g = graph_properties(matrix=a, node_names=rois) 
    c = g.find_communities(levels=7)

    graph_data          = {}
    graph_data['fc']    = f 
    graph_data['adj']   = a 
    graph_data['graph'] = g
    graph_data['comms'] = c
    
    # output
    out_dir = ('/').join(fc_fname.split('/')[:-2])
    fcs = fc_fname.split('/')[-1].split('.xlsx')[0]
    fcs = fcs.replace('roi', atlas)
    out_fname = f'{out_dir}/graphs/{fcs}'
    if weighted: out_fname = f'{out_fname}_weighted_graph.pkl'
    else:        out_fname = f'{out_fname}_unweighted_graph.pkl'
    pickle_file(graph_data, out_fname)


#-------------------------------------------------------------------------------------------
# DEV
#-------------------------------------------------------------------------------------------

## getting mni coords from atlas...
# coords_df = []
# for atlas in ['sub','cortl']:

#     ho = datasets.fetch_atlas_harvard_oxford(atlas + '-maxprob-thr25-2mm')
#     labels = ['HO_' + atlas + '_' + l for l in ho['labels'][1:]]
#     labels = [l.translate(str.maketrans('', '', ''.join([' ', ',', '(', ')', "'", '-', '/']))) for l in labels] # clean up

#     coords = nil.plotting.find_parcellation_cut_coords(ho['maps'], background_label=0, return_label_names=False)
#     df = pd.DataFrame(coords, columns=['MNI_X', 'MNI_Y', 'MNI_Z'])
#     df.insert(0, 'Region_name', labels)
#     coords_df.append(df)
    
# coords_df = pd.concat(coords_df)
# coords_df.to_excel(f'{user}/Desktop/parcellations/HavardOxford_MNI_centers.xlsx', index=False)