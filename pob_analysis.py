# %%
"""
 
 #### ##     ## ########   #######  ########  ########  ######  
  ##  ###   ### ##     ## ##     ## ##     ##    ##    ##    ## 
  ##  #### #### ##     ## ##     ## ##     ##    ##    ##       
  ##  ## ### ## ########  ##     ## ########     ##     ######  
  ##  ##     ## ##        ##     ## ##   ##      ##          ## 
  ##  ##     ## ##        ##     ## ##    ##     ##    ##    ## 
 #### ##     ## ##         #######  ##     ##    ##     ######  
 
"""
%matplotlib qt5

# sys
from collections import OrderedDict
from pathlib import Path
import sys
import os
import dill

# sci
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from tqdm import tqdm

# ephys
import neo
import elephant as ele
import quantities as pq

# plot
import matplotlib.pyplot as plt
import seaborn as sns

# own
import RIOlib as rio
sys.path.append('/home/georg/code/SSSort')
import sssio
from helpers import *

import plotting_globals
from plotting_globals import page_width

"""
 
  ######   #######  ##     ## ##     ##  #######  ##    ## 
 ##    ## ##     ## ###   ### ###   ### ##     ## ###   ## 
 ##       ##     ## #### #### #### #### ##     ## ####  ## 
 ##       ##     ## ## ### ## ## ### ## ##     ## ## ## ## 
 ##       ##     ## ##     ## ##     ## ##     ## ##  #### 
 ##    ## ##     ## ##     ## ##     ## ##     ## ##   ### 
  ######   #######  ##     ## ##     ##  #######  ##    ## 
 
"""

colors = ['#1f77b4','#ff7f0e']
odor_symbols = ['A','B']
odors = ['MHXE','HEPN']
units = ['ab3A','ab3B']

unit_colors = dict(zip(units, colors))
pid_colors = dict(zip(odors, colors))
odor_names = dict(zip(odor_symbols, odors))

save = True
zoom = True

# %%
"""
 
 ######## ##     ##    ###    ##     ## ########  ##       ######## 
 ##        ##   ##    ## ##   ###   ### ##     ## ##       ##       
 ##         ## ##    ##   ##  #### #### ##     ## ##       ##       
 ######      ###    ##     ## ## ### ## ########  ##       ######   
 ##         ## ##   ######### ##     ## ##        ##       ##       
 ##        ##   ##  ##     ## ##     ## ##        ##       ##       
 ######## ##     ## ##     ## ##     ## ##        ######## ######## 
 
"""

pre, post = (-3,2) * pq.s # pob

# high concentration example
path = "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/161202_C/dil2/pob_dil2/sssort_config_pob_dil2.ini"

config_path = Path(path)
blk = get_blk(config_path, pre, post)

# PID read
import configparser
Config = configparser.ConfigParser()
Config.read(config_path)
data_path = Path(Config.get('path', 'data_path'))
letter = data_path.stem.split('_')[2]
pid_path = "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/PID_pob_%s.dill" % letter.upper()

with open(pid_path,'rb') as fH:
    pid_blk = dill.load(fH)

# definition of mapping
PID_map = {'A'     :['A'     ,''],
           'B'     :[''      ,'B'],
           'A_bckg':['A_bckg',''],
           'B_bckg':[''      ,'B_bckg'],
           'AonB'  :['A'     ,'B_bckg'],
           'BonA'  :['A_bckg','B'],
           'Oil'   :[''      ,'' ]}

pid_traces_norm = {}
trial_labels = [seg.annotations['label'] for seg in blk.segments]

for label in np.sort(np.unique(trial_labels)):
    segs = select_by_dict(pid_blk.segments, label=label)
    data = []
    for seg in segs:
        asig = seg.analogsignals[0] # the PID channel
        data.append(asig.magnitude.flatten())
    data = np.array(data)
    data = np.average(data, axis=0)

    # hardcoded baseline determination 
    data = data - np.average(data[:12000]) # 1/2 second
    data = data / data.max()

    # hardcoded mild filtering
    data = np.convolve(data, np.ones(25)/25, mode='same')
    pid_traces_norm[label] = data

pid_traces_norm[''] = np.zeros(data.shape[0])

# %% plot
inspect_labels = ['AonB','BonA']

ymax = blk.segments[0].analogsignals[0].max()
yscl = 1/ymax * 0.75

for label in inspect_labels:

    fig, axes = plt.subplots(nrows=3,
                             sharex=True,
                             gridspec_kw=dict(height_ratios=[1.2,3,4]),
                             figsize=mm2inch(page_width/4, 60))
    
    ### get data
    segs = select_by_dict(blk.segments, label=label)
    segs = calc_firing_rate_for_segs(segs, tau=30)
    
    ### plot stimulus
    # only PID for now
    tvec = segs[0].analogsignals[0].times - segs[0].analogsignals[0].times[0] + pre
    for i, l in enumerate(PID_map[label]):
        pid_trace = pid_traces_norm[l]
        axes[0].plot(tvec, pid_trace, color=pid_colors[odors[i]],
                     lw=0.5, alpha=0.8, label=odors[i])
    
    ### plot data:
    # traces with predicted spikes
    for i, seg in enumerate(segs):
        # the recording
        asig = seg.analogsignals[0]
        tvec = asig.times - asig.times[0] + pre
        axes[1].plot(tvec, asig * yscl + i, lw=0.4, color='k', alpha=0.75)

        # the spiketrains
        for unit in units:
            st, = select_by_dict(seg.spiketrains, unit=unit)
            for t in st.times:
                t_start, t_stop = (t-2*pq.ms, t+2*pq.ms)
                if t_start > asig.t_start and t_stop < asig.t_stop:
                    spike_waveform = asig.time_slice(t-2*pq.ms,t+2*pq.ms)
                    spike_times = spike_waveform.times-asig.times[0] + pre
                    axes[1].plot(spike_times, spike_waveform * yscl + i,
                                 color=unit_colors[unit], lw=0.45, label=unit)

    # plot rates
    rates_all = {}
    for unit in units:
        rates_all[unit] = []

    for i, seg in enumerate(segs):
        for unit in units:
            rate = select_by_dict(seg.analogsignals, kind='firing_rate', unit=unit)[0]
            tvec = rate.times - rate.times[0] + pre
            axes[2].plot(tvec, rate, color=unit_colors[unit], alpha=0.5, lw=0.4)
            rates_all[unit].append(rate.flatten().magnitude)
    
    # plot avg rate
    for unit in units:
        rate_avg = np.average(np.array(rates_all[unit]), 0)
        axes[2].plot(tvec, rate_avg, lw=0.75, alpha=0.9, color=unit_colors[unit],label=unit)

    ### deco
    
    # zoom?
    if 0:
        axes[0].set_xlim([-0.1,0.5])
    else:
        axes[0].set_xlim([-2.5,2])
        
    conc = config_path.stem.split('_')[-1]
        
    fig.suptitle(label)
    
    axes[0].legend(loc='upper right',fontsize='smaller')
    axes[0].set_ylim([-0.1,1.1])
    axes[0].set_ylabel('PID (au)')
    
    # ticks removal
    # axes[0].yaxis.set_major_locator(plt.NullLocator()) # are left intact here because norm
    plt.setp(axes[0].get_xticklabels(), visible=False)
    
    axes[1].yaxis.set_major_locator(plt.NullLocator())
    axes[1].set_ylabel('Recorded\nspikes')
    plt.setp(axes[1].get_xticklabels(), visible=False)        
    
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Spike rate (Hz)')
    
    # removes duplicates
    handles, leg_labels = axes[2].get_legend_handles_labels()
    by_label = OrderedDict(zip(leg_labels, handles))
    axes[2].legend(by_label.values(), by_label.keys(),loc='upper right',fontsize='smaller')
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.04)
    sns.despine(fig)

    #
    for ax in axes:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.4)
        ax.xaxis.set_tick_params(width=0.4)
        ax.yaxis.set_tick_params(width=0.4)
    
    # save output
    if save:
        fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_%s_%s.pdf" % (label, conc))
        fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_%s_%s.svg" % (label, conc))
        fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_%s_%s.png" % (label, conc), dpi=300)

        if zoom:
            axes[0].set_xlim([-0.1,0.5])
            fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_zoom_%s_%s.pdf" % (label, conc))
            fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_zoom_%s_%s.svg" % (label, conc))
            fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_zoom_%s_%s.png" % (label, conc), dpi=300)
    


# %%
"""
 
 ##     ## ######## ########    ###    
 ###   ### ##          ##      ## ##   
 #### #### ##          ##     ##   ##  
 ## ### ## ######      ##    ##     ## 
 ##     ## ##          ##    ######### 
 ##     ## ##          ##    ##     ## 
 ##     ## ########    ##    ##     ## 
 
"""
save = True
pre, post = (-3,2) * pq.s # pob

config_paths = np.genfromtxt("/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/all_pob_results", dtype='U')
concs = ['dil'+path[-5] for path in config_paths]

# read all data
blks = []
for config_path in config_paths:
    config_path = Path(config_path)
    print("reading in %s" % config_path)
    
    blk = get_blk(config_path, pre, post)
    blks.append(blk)

# %% gather firing rate decrease
w = (0.05, 0.55) * pq.s # the window - 50 ms after valve opening (stim delay) + total duration of stimulus
Df = []
for blk_ix, blk in enumerate(blks):
    # barplot with AonB and BonA
    labels = ['AonB','B_bckg','BonA','A_bckg']
    for label in labels:
        segs = select_by_dict(blk.segments, label=label)
        for unit in units:
            for seg in segs:
                st, = select_by_dict(seg.spiketrains, unit=unit)
                n_spikes = st.time_slice(*w + seg.annotations['stim_time']).times.shape[0]
                avgr = n_spikes / (w[1]-w[0]).magnitude
                
                if label == 'A_bckg':
                    if unit == 'ab3A':
                        label_mod = 'cognate only'
                    if unit == 'ab3B':
                        label_mod = 'unused'
                        
                if label == 'B_bckg':
                    if unit == 'ab3B':
                        label_mod = 'cognate only'
                    if unit == 'ab3A':
                        label_mod = 'unused'
                        
                if label == 'BonA':
                    if unit == 'ab3B':
                        label_mod = 'unused'
                    if unit == 'ab3A':
                        label_mod = 'plus incognate'
                        
                if label == 'AonB':
                    if unit == 'ab3A':
                        label_mod = 'unused'
                    if unit == 'ab3B':
                        label_mod = 'plus incognate'
                
                recording_name = Path(config_paths[blk_ix]).parts[-4]
                S = pd.Series(dict(recording=recording_name,conc=concs[blk_ix],label=label_mod,unit=unit,rate=avgr))
                Df.append(S)

Df = pd.DataFrame(Df)
Df.reset_index()

# %% trial averaging
Df = Df.groupby(['recording','conc','label','unit']).mean()
Df = Df.reset_index()

# %% normalization
# normalize to cognate only, per animal
for i, rec_group in Df.groupby(['recording','conc']):
    for unit in units:
        sub_group = rec_group.groupby('unit').get_group(unit)
        cogn_rate = sub_group.groupby('label').get_group('cognate only')['rate'].values
        Df.loc[sub_group.index,'rate_norm'] = rec_group.groupby('unit').get_group(unit)['rate'].values / cogn_rate
                                            

# %%
rate_key = 'rate'

# # drop all unused
# Df = Df.loc[Df['label'] != 'unused']
# Df['rate'] = Df['rate'].astype('float32')

colors = sns.color_palette('muted',n_colors=5)
colors = sns.color_palette(np.array(colors)[[2,4],:])
# colors = sns.color_palette('Set2', n_colors=2)


fig, axes = plt.subplots(ncols=2, sharey=True, figsize=mm2inch(page_width/3,60))
                        #  figsize=helpers.mm2inch((koma_width/2,
                        #                           koma_width/2)))


cgroups = Df.groupby('conc')
data_dil3 = cgroups.get_group('dil3').groupby(['recording','unit','label'], as_index=False)[rate_key].mean()
data_dil2 = cgroups.get_group('dil2').groupby(['recording','unit','label'], as_index=False)[rate_key].mean()

kwargs = dict(x='unit', y=rate_key, hue='label', hue_order=['cognate only','plus incognate'], order=['ab3A','ab3B'], palette=colors)
sns.barplot(data=data_dil3, ax=axes[0], **kwargs)
sns.barplot(data=data_dil2, ax=axes[1], **kwargs)

[ax.legend().remove() for ax in axes]
axes[1].legend(loc='upper right',fontsize='small')
axes[1].set_ylabel('')

if rate_key == 'rate_norm':
    axes[0].set_ylabel('Normalized spike rate')
else:
    axes[0].set_ylabel('Spike rate (Hz)')

axes[0].set_title('Low concentration')
axes[1].set_title('High concentration')

axes[0].set_xlabel('')
axes[1].set_xlabel('')

# fig.suptitle('Response reduction by incognate odorant')

# plot lines between the individual measurement points
# iterate over concentrations
supergroup = Df.groupby(['recording','conc','unit'])
all_keys = supergroup.groups.keys()
for key in all_keys:
    sub = supergroup.get_group(key)
    y1 = np.average(sub[sub['label'] == 'cognate only'][rate_key])
    y2 = np.average(sub[sub['label'] == 'plus incognate'][rate_key])
    w = 0.20
    
    if key[1] == 'dil3':
        ax = axes[0]
    else:
        ax = axes[1]

    if key[2] == 'ab3A':
        pos = [0-w,0+w]
    else:
        pos = [1-w,1+w]
    
    ax.plot(pos,[y1,y2],color='k', alpha=0.5, lw=1)

# making room for legend and stars
y1,y2 = ax.get_ylim()
ax.set_ylim(y1,y1+(y2-y1)*1.20)    
kwargs = dict(ha='center',fontsize='large')

# paired t-tests
yoffs = 62

groups = Df.groupby(['conc','label','unit'])
concs = ['dil3','dil2']
for j, conc in enumerate(concs):
    for i, unit in enumerate(units):
        a = groups.get_group((conc,'cognate only', unit))[rate_key].values
        b = groups.get_group((conc,'plus incognate',unit))[rate_key].values
        p = (stats.ttest_rel(a,b))[1]
        print("conc: %s, unit: %s, p=%.6f" % (conc, unit, p))
        axes[j].annotate(p2stars(p), xy=(i, yoffs), **kwargs)

fig.tight_layout()
# fig.subplots_adjust(top=0.87)

for ax in axes:
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.4)
    ax.xaxis.set_tick_params(width=0.4)
    ax.yaxis.set_tick_params(width=0.4)

# save output
if save:
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_summary.pdf")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_summary.svg")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_summary.png", dpi=300)


# %%
