import pandas as pd
import numpy as np
import datetime

# IIP : simple mean (0-4)

def reverse_score(scores_df, high_score, rev_items):
    '''Reverse scores: e.g., 1,2,3,4 -> 4,3,2,1
         high_score = highest score available
         rev_items = 0-ixed array of items to reverse 
         returns: full array of all values, easy to split up into subscales etc
     '''
    scores = scores_df.values
    if high_score > 1:
        scores[:,rev_items] = (high_score+1) - scores[:,rev_items]
    elif high_score == 1:    
        scores[:,rev_items] = (scores[:,rev_items] == 0)*1
    return pd.DataFrame(scores, columns=scores_df.columns.values)

def score_sni(df):

    sni_cols = [c for c in df.columns if 'sni' in c]
    sni_cols = [c for c in sni_cols if c != 'sni_timestamp']
    sni_items = df[sni_cols]
    
    sni_items = sni_items.fillna(0)
    sni_scores = np.zeros((len(df),3))
    
    # number of high-contact roles
    sni_scores[:,0] += (sni_items.loc[:,'sni_1'] == 1) * 1
    for item in ['2a','3a','4a','5a','6a','7a','8a','10','11a']:
        sni_scores[:,0] += (sni_items.loc[:,'sni_'+item] > 0) * 1
    for n in np.arange(1,8):
        sni_scores[:,0] += (sni_items.loc[:,'sni_12a' + str(n) + 'number'] > 1) * 1
    sni_scores[:,0] += (sni_items.loc[:, 'sni_9a'] > 0) & (sni_items.loc[:, 'sni_9b'] > 0) * 1

    # number of people in social network
    sni_scores[:,1] += (sni_items.loc[:,'sni_1'] == 1) * 1
    sni_scores[:,1] += sni_items.loc[:,'sni_2a']
    sni_items.loc[:,'sni_3a'][sni_items.loc[:,'sni_3a'] == 3] = 2
    sni_items.loc[:,'sni_3a'][((sni_items.loc[:,'sni_3a'] == 1) | (sni_items.loc[:,'sni_3a'] == 2))] = 1 # 1 or 2 -> 1
    sni_items.loc[:,'sni_4a'][sni_items.loc[:,'sni_4a'] == 3] = 2
    sni_items.loc[:,'sni_4a'][((sni_items.loc[:,'sni_4a'] == 1) | (sni_items.loc[:,'sni_4a'] == 2))] = 1 # 1 or 2 -> 1
    for item in ['3a','4a','5a','9a','9b','10','11a']:
        sni_scores[:,1] += sni_items.loc[:,'sni_'+item]
    for n in np.arange(1,8):
        sni_scores[:,1] += sni_items.loc[:,'sni_12a' + str(n) + 'number']

    # number of embedded networks
    sni_scores[:,2] += (np.sum(sni_items.loc[:,['sni_1','sni_2a','sni_3a','sni_4a','sni_5a']], axis=1) > 4) * 1
    for item in ['6a','7a','8a','10','11a']:
        sni_scores[:,2] += (sni_items.loc[:,'sni_'+ item] > 4)
    sni_scores[:,2] += (sni_items.loc[:,'sni_9a'] + sni_items.loc[:,'sni_9b'] > 4) * 1
    for n in np.arange(1,8):
        sni_scores[:,2] += (sni_items.loc[:,'sni_12a' + str(n) + 'number'] > 4) * 1

    # output
    sni_scores = pd.DataFrame(sni_scores, columns=['sni_hc_score', 'sni_size_score', 'sni_diversity_score'])
    sni_df = pd.concat([sni_items, sni_scores], axis=1)
    return sni_df

def score_pid(df):
    
    neg_aff = np.array([8,9,10,11,15])-1
    detach = np.array([4,13,14,16,18])-1
    antagonism = np.array([17,19,20,22,25])-1
    disinhibtion = np.array([1,2,3,5,6])-1
    psychotism = np.array([7,12,21,23,24])-1

    pid_df = df[['pid5_' + str(i) for i in range(1,26)]]
    pid_df['pid5_neg_aff_score'] = np.sum(pid_df.iloc[:,neg_aff], axis=1)
    pid_df['pid5_detachment_score'] = np.sum(pid_df.iloc[:,detach], axis=1)
    pid_df['pid5_antagonism_score'] = np.sum(pid_df.iloc[:,antagonism], axis=1)
    pid_df['pid5_disinhibiton_score'] = np.sum(pid_df.iloc[:,disinhibtion], axis=1)
    pid_df['pid5_psychotism_score'] = np.sum(pid_df.iloc[:,psychotism], axis=1)
    
    return pid_df

def score_bsl(df):
    
    bsl_df = df[['bsl_' + str(n) for n in range(1, 24)]]
    bsl_df = score(bsl_df)
    bsl_df['bsl_score'] = bsl_df['bsl_score']/23
    bsl_df['bsl_%'] = df['bsl_24']
    
    return bsl_df

def score(ques):
    ''' 
        generic sum, excluding nan responses
    '''
    num_q = ques.shape[1]
    score_name = ques.columns[0].replace('1', 'score')
    ques.insert(num_q, score_name, np.sum(ques, axis=1))
    ques.iloc[np.array(np.sum(np.isfinite(ques), axis=1) == 1), num_q] = np.nan # ensure non-responses are kept that way
    return ques

def get_sleep_times(ques_df):
    
    sleep_cols  = [c for c in ques_df.columns if 'sleep' in c] +  \
                  [c for c in ques_df.columns if 'bedtime' in c] + \
                  [c for c in ques_df.columns if 'wakeup' in c]
    sleep_times = ques_df[sleep_cols].fillna(0)
    sleep_times[['am_pm_bedtime','am_pm_wakeup']] = sleep_times[['am_pm_bedtime','am_pm_wakeup']].replace(to_replace=2, value='PM')
    sleep_times[['am_pm_bedtime','am_pm_wakeup']] = sleep_times[['am_pm_bedtime','am_pm_wakeup']].replace(to_replace=1, value='AM')
    
    sleep = []
    for s, sub in sleep_times.iterrows():
        
        # bedtime
        bed_hr    = int(sub['hour_bedtime'])
        bed_min   = int(sub['minutes_bedtime'])
        bed_ampm  = sub['am_pm_bedtime']

        # wakeup
        wake_hr   = int(sub['hour_wakeup'])
        wake_min  = int(sub['minutes_wakeup'])
        wake_ampm = sub['am_pm_wakeup']

        # one weird number... check redcap
        if len(str(int(wake_hr))) > 3: 
            wake_min = int(str(wake_hr)[2:4])
            wake_hr = int(str(wake_hr)[0:2])

        # clean up: 
        # bed is pm, wake is am - diff days (eg, 11pm 7am)
        if bed_hr == 12: bed_hr = 0
        if (bed_ampm == 'PM') & (wake_ampm == 'AM'):
            if bed_hr == 12: 
                bed_day = 2
            else:
                bed_day = 1
                bed_hr  = bed_hr + 12

        # bed is am, wake is pm - same day (eg, 1am 1pm)
        elif (bed_ampm == 'AM') & (wake_ampm == 'PM'):
            bed_day = 2
            if wake_hr != 12:
                wake_hr = wake_hr + 12

        # both are pm - diff days (eg, 11pm 1pm)
        elif (bed_ampm == 'PM') & (wake_ampm == 'PM'):
            bed_day = 1
            wake_hr = wake_hr + 12

        # both are am - same day (eg, 1am 9am)
        elif (bed_ampm == 'AM') & (wake_ampm == 'AM'):
            bed_day = 2 
                
    #         if bed_hr == 11: # WEIRD - flag it
    #             bed_hr = 0 # round it up

        # get time difference
        try: 
            bedtime  = datetime.datetime(2022, 1, bed_day,
                                         int(bed_hr), int(bed_min), 0)
            waketime = datetime.datetime(2022, 1, 2,
                                         int(wake_hr), int(wake_min), 0)   
            sleep.append((waketime-bedtime).total_seconds() / 3600) # 3600 seconds in an hour
        except: # weird entries
            sleep.append(np.nan)
            
    sleep_times['sleep_hours'] = sleep
    return sleep_times

def score_bapq(df):
    bapq_rev = np.array([1,3,7,9,12,15,16,19,21,23,25,28,30,34,36])-1
    bapq_df_ = subset_df(df, ['bapq'])
    bapq_df = reverse_score(bapq_df_, 6, bapq_rev)
    bapq_df['bapq_score'] = np.sum(bapq_df, 1).values
    bapq_df['bapq_aloof_score'] = np.sum(bapq_df.iloc[:,np.array([1,5,9,12,16,18,23,25,27,28,31,36])-1],1).values
    bapq_df['bapq_prag_lang_score'] = np.sum(bapq_df.iloc[:,np.array([2,4,7,10,11,14,17,20,21,29,32,34])-1],1).values
    bapq_df['bapq_rigid_score'] = np.sum(bapq_df.iloc[:,np.array([3,6,8,13,15,19,22,24,26,30,33,35])-1],1).values
    
    return bapq_df

def score_sss(df):
    
    sss_items = ['sss_' + str(i) for i in range(1,44)]
    sss_df = reverse_score(df[sss_items].copy(), 1, sss_rev)
    
    # summarize
    sss_df['sss_att'] = df['sss_att']
    sss_df['sss_score'] = np.sum(sss_df, axis=1)
    sss_df['sss_unus_exp_score'] = np.sum(sss_df.iloc[:,np.arange(1,13)-1],axis=1)
    sss_df['sss_cog_dis_score'] = np.sum(sss_df.iloc[:,np.arange(13,24)-1],axis=1)
    sss_df['sss_intro_anhe_score'] = np.sum(sss_df.iloc[:,np.arange(24,34)-1],axis=1)
    sss_df['sss_impuls_noncon_score'] = np.sum(sss_df.iloc[:,np.arange(34,44)-1],axis=1)
    
    return sss_df

def score_eat(df):

    eat_items = ['eat_' + str(i) for i in range(1,27)]
    dieting_items = np.array([1,6,7,10,11,12,14,16,17,22,23,24,26])-1
    preoccup_items = np.array([3,4,9,18,21,25])-1
    control_items = np.array([2,5,8,13,15,19,20])-1

    eat_df = df[eat_items].copy()
    
    # rescore stuff
    eat_df[eat_df.iloc[:,0:25] > 3] = 0
    eat_df[eat_df.iloc[:,0:25] == 1] = 3
    eat_df[eat_df.iloc[:,0:25] == 3] = 1
    eat_df[eat_df.iloc[:,25] < 3] = 0
    eat_df[eat_df.iloc[:,25] == 4] = 1
    eat_df[eat_df.iloc[:,25] == 5] = 2
    eat_df[eat_df.iloc[:,25] == 6] = 3
    
    # summarize
    eat_df['eat_score'] = np.sum(eat_df.iloc[:,dieting_items],axis=1)
    eat_df['eat_dieting_score'] = np.sum(eat_df.iloc[:,dieting_items],axis=1)
    eat_df['eat_bulmia_food_preoc_score'] = np.sum(eat_df.iloc[:,dieting_items],axis=1)
    eat_df['eat_oral_control_score'] = np.sum(eat_df.iloc[:,control_items],axis=1)
    
    return eat_df

def score_pid(df):
    
    neg_aff = np.array([8,9,10,11,15])-1
    detach = np.array([4,13,14,16,18])-1
    antagonism = np.array([17,19,20,22,25])-1
    disinhibtion = np.array([1,2,3,5,6])-1
    psychotism = np.array([7,12,21,23,24])-1

    pid_df = df[['pid5_' + str(i) for i in range(1,26)]]
    pid_df['pid5_neg_aff_score'] = np.sum(pid_df.iloc[:,neg_aff], axis=1)
    pid_df['pid5_detachment_score'] = np.sum(pid_df.iloc[:,detach], axis=1)
    pid_df['pid5_antagonism_score'] = np.sum(pid_df.iloc[:,antagonism], axis=1)
    pid_df['pid5_disinhibiton_score'] = np.sum(pid_df.iloc[:,disinhibtion], axis=1)
    pid_df['pid5_psychotism_score'] = np.sum(pid_df.iloc[:,psychotism], axis=1)
    
    return pid_df

def score_ctq(df):
 
    ''' 
        All five abuse and neglect subscales are sums of the scorings from ‘never true’ (score 1) to ‘very often true’ (score 5), 
        and after reversing seven items, all subscales can therefore vary between 5 and 25. 
        
        The M/D scale is different, since only the highest positive scores (score 5) are counted, and it can therefore vary from 0 to 3.
    '''
    
    # get items
    subscales = {'emotional_abuse':   np.array([3,8,14,18,25])-1, 
                 'emotional_neglect': np.array([5,7,13,19,28])-1,
                 'physical_abuse':    np.array([9,11,12,15,17])-1, 
                 'physical_neglect':  np.array([1,2,4,6,26])-1, 
                 'sexual_abuse':      np.array([20,21,23,24,27])-1}    
    ctq_data = df[[c for c in df.columns if ('ctq' in c) & 
                   (c not in ['ctq_timestamp','ctq_complete'])]]
    ctq_data = reverse_score(ctq_data, 5, np.array([2,5,7,13,19,26,28])-1) 

    # score
    for scale, items in subscales.items():
        ctq_data['ctq_' + scale + '_score'] = np.sum(ctq_data.iloc[:, items], 1)
    ctq_data['ctq_total_score'] = np.sum(ctq_data[[f'ctq_{s}_score' for s in subscales.keys()]], 1)
    ctq_data['ctq_minimization_score'] = np.sum(ctq_data.iloc[:, np.array([10,16,22])-1] == 5.0, 1)
    
    return ctq_data

def score_mos(mos_raw):
    '''
        score the mos social support survey
        pass in array with raw mosss responses in columns
        mos details:
            - multiple subscales + overall mean
            - no reverse scoring
            - item 13 not in a subscale but is in overall summary
    '''
    # check that there are 19 columns
    if mos_raw.shape[1] != 19:
        raise Exception(f'Input not 19 columns: {mos_raw.shape}')

    subscales = {'emot_info_supp': np.array([2,3,7,8,12,15,16,18])-1, 
                'tangible_supp': np.array([1,4,11,14])-1,
                'affect_supp': np.array([5,9,19])-1, 
                'pos_social_inter': np.array([6,10,17])-1}    

    # subscale means
    mos_scores = []
    for _, ixs in subscales.items():
        mos_scores.append(np.nanmean(mos_raw[:, ixs], axis=1))
    mos_scores.append(np.nanmean(mos_raw, axis=1)) # overall mean
    mos_scores = np.vstack(mos_scores).T

    return pd.DataFrame(mos_scores, 
                        columns=[f'mossss_{scale}_score' for scale in subscales.keys()] + ['mossss_overall_score'])