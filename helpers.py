import os, sys
import numpy as np
import pandas as pd
import neo
sys.path.append('/home/georg/code/SSSort')
import sssio
import quantities as pq
import elephant as ele

def p2stars(p, ns=False):
    """ turns a p-value into significance stars with the
    classic alpha levels 0.05, 0.005, 0.0005 """
    if ns:
        s = 'n.s.'
    else:
        s = ''
    if p < 0.05:
        s = '*'
    if p < 0.005:
        s += '*'
    if p < 0.0005:
        s += '*'
    return s

def select_by_dict(objs,**selection):
    """ returns all where dict entries are matched 
    http://stackoverflow.com/questions/30818694/test-if-dict-contained-in-dict
    """
    res = []
    for obj in objs:
        if selection.items() <= obj.annotations.items():
            res.append(obj)
    return res

def get_unit_names(csv_path):
    #  read unit names from file
    df = pd.read_csv(csv_path, names=['sssort','dros'])
    k = df['sssort'].values.astype('U')
    v = df['dros'].values
    unit_names = dict(zip(k,v))
    return unit_names

def split_block(blk, trial_times, trial_labels, pre, post):
    # make new
    blk_new = neo.core.Block()

    # copy annotations
    blk_new.annotate(**blk.annotations)

    # slice into trials
    for t, label in zip(trial_times, trial_labels):
        seg = blk.segments[0].time_slice(t+pre,t+post)
        seg.annotations['label'] = label
        seg.annotations['stim_time'] = t
        blk_new.segments.append(seg)

    return blk_new

def FDR_correct(S, nComp=None):
    """
    Benjamini Hochberg procedure == false discovery rate correction
    adjusts p-values for multiple comparisons

    takes: series of p values, returns series with stars from FDR correction 

    nComp is the number of comparisons """

    # sort ascending 
    S = S.sort_values(ascending=False)
    F = pd.Series(['']*S.shape[0], index=S.index)
    
    if nComp is None:
        nComp = S.shape[0]
    
    for i,label in enumerate(S.index):
        if S.loc[label] < (i+1) * (0.05/nComp):
            F.loc[label:] = '*'
        if S.loc[label] < (i+1) * (0.005/nComp):
            F.loc[label:] = '**'
        if S.loc[label] < (i+1) * (0.0005/nComp):
            F.loc[label:] = '***'
    
    return F

def get_blk(config_path, pre, post):
    """ helper to read SSSort output """
    if (config_path.parent / "results_manual").exists():
        print('using manually curated results')
        results_folder = config_path.parent / "results_manual"
    else:
        results_folder = config_path.parent / "results"

    blk_path = results_folder / "result.dill"

    blk_full  = sssio.dill2blk(blk_path)

    # trial labels path inference
    exp = blk_path.parts[-3].split('_')[0]
    letter = blk_path.parts[-5].split('_')[1]
    candidates = os.listdir(blk_path.parent.parent.parent)
    trial_labels_path = blk_path.parent.parent.parent / [c for c in candidates if exp in c and c.endswith('trial_labels_corr')][0]

    trial_labels = np.genfromtxt(trial_labels_path, dtype='U')
    labels_unique = np.sort(trial_labels)

    # reading SSR data
    blk_full = sssio.dill2blk(blk_path)

    # read unit names from file and assign to spike trains
    unit_names = get_unit_names(blk_path.parent / 'unit_names.csv')

    # rename all spiketrains
    for st in blk_full.segments[0].spiketrains:
        if 'unit' in st.annotations:
            st.annotations['unit'] = unit_names[st.annotations['unit']]

    # rename all firing rates
    for asig in blk_full.segments[0].analogsignals:
        if 'unit' in asig.annotations:
            asig.annotations['unit'] = unit_names[asig.annotations['unit']]

    # split blk with one long segment into many segments, corresponding to trials
    trial_times = blk_full.segments[0].events[0].times
    blk = split_block(blk_full, trial_times, trial_labels, pre, post)
    return blk

def calc_firing_rate_for_segs(segs, tau=50):
    """ calculate alphakernel based firing rate """
    kernel = ele.kernels.AlphaKernel(tau*pq.ms)
    for seg in segs:
        for st in seg.spiketrains:
            if 'unit' in st.annotations:
                if st.annotations['unit'] != 'trash':
                    ft = 1/seg.analogsignals[0].sampling_rate
                    rate = ele.statistics.instantaneous_rate(st, t_start = st.t_start, sampling_period=ft, kernel=kernel)[:,0]
                    rate.t_start = st.t_start

                    rate.annotate(**st.annotations)
                    rate.annotate(kind='firing_rate')
                    seg.analogsignals.append(rate)

    return segs

def mm2inch(*args):
    """ helper for plotting """
    return [arg/25.4 for arg in args]