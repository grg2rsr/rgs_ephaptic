# %%
import neo
from neo import Spike2IO
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from pathlib import Path
import dill

# %% 
"""
 
 ########  ########  ######  
 ##     ##    ##    ##    ## 
 ##     ##    ##    ##       
 ##     ##    ##     ######  
 ##     ##    ##          ## 
 ##     ##    ##    ##    ## 
 ########     ##     ######  
 
"""
smr_paths = ["/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/raw/161208_A_dts.smr",
             "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/raw/161208_B_dts.smr",
             "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/raw/161208_C_dts.smr"]

labels_paths = ["/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_11_28_HEPN_MHXE_dts_3_6_12_24_48_96_a/trial_labels.trial_labels_corr",
                "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_11_28_HEPN_MHXE_dts_3_6_12_24_48_96_b/trial_labels.trial_labels_corr",
                "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_11_28_HEPN_MHXE_dts_3_6_12_24_48_96_c/trial_labels.trial_labels_corr"]

twice = False
tcut = (-1,2) * pq.s # dts
name = 'dts'
description = 'PID traces for dts experiment'
outlabel = "PID_dts_x.dill"

# %%
"""
 
 ######## ##       ##     ##  ######  ######## 
 ##       ##       ##     ## ##    ##    ##    
 ##       ##       ##     ## ##          ##    
 ######   ##       ##     ## ##          ##    
 ##       ##       ##     ## ##          ##    
 ##       ##       ##     ## ##    ##    ##    
 ##       ########  #######   ######     ##    
 
"""
smr_paths = ["/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/raw/161208_A_fluct_2x.smr",
             "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/raw/161208_B_fluct_2x.smr",
             "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/raw/161208_C_fluct_2x.smr"]

labels_paths = ["/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_11_28_HEPN_MHXE_fluct_tau50_a/trial_labels.trial_labels_corr",
                "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_11_28_HEPN_MHXE_fluct_tau50_b/trial_labels.trial_labels_corr",
                "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_11_28_HEPN_MHXE_fluct_tau50_c/trial_labels.trial_labels_corr"]

twice = True
tcut = (-1,12) * pq.s # fluct
name = 'fluct'
description = 'PID traces for fluct'
outlabel = "PID_fluct_x.dill"

# %%
"""
 
 ########   #######  ########  
 ##     ## ##     ## ##     ## 
 ##     ## ##     ## ##     ## 
 ########  ##     ## ########  
 ##        ##     ## ##     ## 
 ##        ##     ## ##     ## 
 ##         #######  ########  
 
"""
smr_paths = ["/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/raw/161208_A_pob_2x.smr",
             "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/raw/161208_B_pob_2x.smr",
             "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/raw/161208_C_pob_2x.smr"]

labels_paths = ["/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_12_01_HEPN_MHXE_pulse_on_background_corrected_a/trial_labels.trial_labels_corr",
                "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_12_01_HEPN_MHXE_pulse_on_background_corrected_b/trial_labels.trial_labels_corr",
                "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_12_01_HEPN_MHXE_pulse_on_background_corrected_c/trial_labels.trial_labels_corr"]

twice = True
tcut = (-3,2) * pq.s # pob
name = 'pob'
description = 'PID traces for pulse on background experiment'
outlabel = "PID_pob_x.dill"

# %%
"""
 
 ########  ##     ## ##    ## 
 ##     ## ##     ## ###   ## 
 ##     ## ##     ## ####  ## 
 ########  ##     ## ## ## ## 
 ##   ##   ##     ## ##  #### 
 ##    ##  ##     ## ##   ### 
 ##     ##  #######  ##    ## 
 
"""

fs = 25*pq.kHz
dt = (1/fs).simplified
reltime = np.arange(tcut[0].magnitude,tcut[1].magnitude,dt.magnitude)
letters = ['A','B','C']
ouput_folder = Path("/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/")

for i in range(3):
    labels = np.genfromtxt(labels_paths[i],dtype='U')
    if twice:
        labels = np.concatenate([labels,labels])

    # read data
    blk = neo.core.Block(name=name, description=description)
    reader = Spike2IO(smr_paths[i])
    segment = reader.read_segment()

    # extract data
    pid = segment.analogsignals[0].magnitude.flatten()
    tvec = segment.analogsignals[0].times.magnitude
    data = np.stack([tvec,pid]).T
    
    # npy data dump
    fname = outlabel.replace('x', letters[i])
    outpath = (ouput_folder / fname).with_suffix('.npy')
    np.save(outpath, data)

    stim_times = segment.events[0].times
    np.save(outpath.parent / (outpath.stem + '_stim_times.npy'), stim_times.magnitude)
    
    # cut into trials
    for j, t in enumerate(stim_times):
        seg = segment.time_slice(t+tcut[0], t+tcut[1])
        seg.annotate(label=labels[j], stim_time=t)
        blk.segments.append(seg)

    # save it to disk as dill
    outpath = ouput_folder / fname
    with open(outpath,'wb') as fH:
        dill.dump(blk,fH)

# %%
