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
sys.path.append('/home/georg/code/SSSort') # manually appending sssort installation
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

save = True
zoom = False
pre, post = (-0.5, 1.25) * pq.s

# get ephys data
config_path = Path("/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/161202_A/dil3/dts_dil3/sssort_config_dts_dil3.ini")
# config_path = Path("/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/161205_D/dil2/dts_dil2/sssort_config_dts_dil2.ini")
blk = get_blk(config_path, pre, post)

# get rio sequence
import configparser
Config = configparser.ConfigParser()
Config.read(config_path)
data_path = Path(Config.get('path', 'data_path'))
letter = data_path.stem.split('_')[2]

sqc_path = "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_11_28_HEPN_MHXE_dts_3_6_12_24_48_96_%s/HEPN_MHXE_dts_3_6_12_24_48_96_templates_corr.sqc" % letter.lower()
rio_seq = rio.read_sqc(sqc_path)

inspect_labels = ['A','AB','B']

ymax = blk.segments[0].analogsignals[0].max()
yscl = 1/ymax * 0.75

for label in inspect_labels:

    fig, axes = plt.subplots(nrows=3,
                             sharex=True,
                             gridspec_kw=dict(height_ratios=[1.4,3,4]),
                             figsize=mm2inch(page_width/3, 60))
    
    ### get data
    segs = select_by_dict(blk.segments, label=label)
    segs = calc_firing_rate_for_segs(segs)
    
    ### plot stimulus
    tvec = segs[0].analogsignals[0].times - segs[0].analogsignals[0].times[0] + pre

    # sqc pattern
    ix = [Pattern.name for Pattern in rio_seq.Patterns].index(label)
    Pattern = rio_seq.Patterns[ix]

    ch_flip = {0:1,1:0}

    for i in range(2):
        j = ch_flip[i]
        ch = Pattern.calc_states(j).astype('int32')
        n_pre = (np.absolute(pre).magnitude * 1000).astype('int32')
        n_post = (np.absolute(post - 0.5*pq.s).magnitude * 1000).astype('int32')
        ch = np.pad(ch,(n_pre, n_post), mode='constant', constant_values=0)
        tvec_p = np.linspace(tvec[0],tvec[-1],ch.shape[0])
        axes[0].plot(tvec_p, ch*0.8 + i*1, color=pid_colors[odors[i]],
                     alpha=0.8, lw=0.5, label=odors[i])

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
            rate, = select_by_dict(seg.analogsignals, kind='firing_rate', unit=unit)
            tvec = rate.times - rate.times[0] + pre
            axes[2].plot(tvec, rate, color=unit_colors[unit], alpha=0.5, lw=0.4)
            rates_all[unit].append(rate.flatten().magnitude)
    
    # plot avg rate
    for unit in units:
        rate_avg = np.average(np.array(rates_all[unit]), 0)
        axes[2].plot(tvec, rate_avg ,lw=0.75, alpha=0.9,color=unit_colors[unit],label=unit)

    ### deco
    fig.suptitle(label)
    
    axes[0].legend(loc='upper right')
    axes[0].set_ylim([-0.2,2])
    axes[0].set_ylabel('Valve\nstate')
    axes[0].yaxis.set_major_locator(plt.NullLocator())
    
    # ticks removal
    # axes[0].yaxis.set_major_locator(plt.NullLocator()) # are left intact here because norm
    plt.setp(axes[0].get_xticklabels(), visible=False)
    
    axes[1].yaxis.set_major_locator(plt.NullLocator())
    axes[1].set_ylabel('Voltage\ntraces')
    plt.setp(axes[1].get_xticklabels(), visible=False)        
    
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Spike rate (1/s)')
    
    # removes duplicates
    handles, leg_labels = axes[1].get_legend_handles_labels()
    by_label = OrderedDict(zip(leg_labels, handles))
    axes[2].legend(by_label.values(), by_label.keys(),loc='upper right')
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.04, top=0.9)

    for ax in axes:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.4)
        ax.xaxis.set_tick_params(width=0.4)
        ax.yaxis.set_tick_params(width=0.4)


    # save output
    if save:
        fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_example_%s.pdf" % label)
        fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_example_%s.svg" % label)
        fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_example_%s.png" % label, dpi=300)

        # zoom
        if zoom:
            axes[0].set_xlim([-0.1,0.5])
            fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_example_zoom_%s.pdf" % label)
            fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_example_zoom_%s.svg" % label)
            fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_example_zoom_%s.png" % label, dpi=300)


    # # save output
    # if save:
    #     fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_%s_%s.pdf" % (label, conc))
    #     fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_%s_%s.svg" % (label, conc))
    #     fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_%s_%s.png" % (label, conc), dpi=300)

    #     # zoom?
    #     axes[0].set_xlim([-0.1,0.5])
    #     fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_zoom_%s_%s.pdf" % (label, conc))
    #     fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_zoom_%s_%s.svg" % (label, conc))
    #     fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/pob/pob_example_zoom_%s_%s.png" % (label, conc), dpi=300)
    

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

path = Path("/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/all_dts_dil3_results")
config_paths = np.genfromtxt(path,dtype='U')
pre, post = (-0.5, 1.25) * pq.s

# get all data
blks = []
for config_path in config_paths:
    config_path = Path(config_path)
    results_folder = config_path.parent  / 'results'
    print(results_folder)

    blk = get_blk(config_path, pre, post)
    blks.append(blk)

# %% extract all data - average over trials
S = []
for blk_ind, blk in enumerate(blks):
    trial_labels = [seg.annotations['label'] for seg in blk.segments]

    for seg in blk.segments:
        for unit in units:
            asig, = select_by_dict(seg.analogsignals, unit=unit, kind='frate_fast')
            frate_max = asig.magnitude.max()
            label = seg.annotations['label']
            recording = Path(blk.segments[0].annotations['filename']).parts[-3]
            series = pd.Series(dict(recording=recording, stim=label, rate=frate_max, unit=unit))
            S.append(series)

Df = pd.DataFrame(S)

# trial averaging
Df = Df.groupby(['stim','unit','recording']).mean() 
Df = Df.reset_index()

# %% other metrics - calculated on a trial avg basis
kernel = ele.kernels.AlphaKernel(50*pq.ms)
ft = 1/seg.analogsignals[0].sampling_rate

TimeDf = []
for blk_ind, blk in enumerate(blks):
    trial_labels = [seg.annotations['label'] for seg in blk.segments]
    for unit in units:
        for label in np.unique(trial_labels):
            S = []
            segs = select_by_dict(blk.segments, label=label)
            for seg in segs:
                st, = select_by_dict(seg.spiketrains, unit=unit)
                rate = ele.statistics.instantaneous_rate(st, t_start = st.t_start, sampling_period=ft, kernel=kernel)[:,0]
                S.append(rate.magnitude.flatten())
            
            min_shape = np.min([s.shape[0] for s in S])
            S = np.array([s[:min_shape] for s in S])
            s = np.average(S, axis=0)

            tvec = np.linspace(pre.magnitude,post.magnitude,s.shape[0]) * pq.s
            b = np.average(s[tvec < 0])
            fmax = np.max(s)
            t_ix = np.argmin(np.absolute(tvec)) + np.argmax(s[tvec > 0] > (b + (fmax - b) * 1/np.exp(1)))
            t_onset = np.float32(tvec[t_ix].magnitude)
            t_peak = np.float32(tvec[np.argmax(s)].magnitude)

            # mod peak
            if label not in ['A','B','AB','Air']:
                if unit == 'ab3A':
                    if label.startswith('B'):
                        t_corr = np.float32(label[1:-1])/1e3
                    else:
                        t_corr = 0
                if unit == 'ab3B':
                    if label.startswith('A'):
                        t_corr = np.float32(label[1:-1])/1e3
                    else:
                        t_corr = 0
                t_peak = t_peak - t_corr
                t_onset = t_onset - t_corr

            recording = Path(blk.segments[0].annotations['filename']).parts[-3]
            TimeDf.append(pd.Series(dict(stim=label, unit=unit, recording=recording, onset=t_onset*1e3, peak_time=t_peak)))
TimeDf = pd.DataFrame(TimeDf)

Df = pd.merge(Df, TimeDf,on=['stim','unit','recording'])

# %% # drop trash
Df = Df.loc[Df['unit'] != 'trash']

# normalizations
units = ['ab3A','ab3B']
recordings = Df['recording'].unique()
cognate_odors = dict(ab3A='A', ab3B='B')

# normalize within animal:
# all firing rates are divided by the median rate of the units cognate odorant
# SEPERATELY for each animal
# -> all cognate responses are set to 1, with 0 sd
for i, rec_group in Df.groupby('recording'):
    for unit in units:
        unit_rate = rec_group.groupby(['unit','stim']).get_group((unit, cognate_odors[unit]))['rate']
        rec_unit_group = rec_group.groupby('unit').get_group(unit)
        Df.loc[rec_unit_group.index, 'rate_norm'] = rec_unit_group['rate'].values / unit_rate.values

# normalize across animals:
# all firing rates are divided by the median rate of the units cognate odorant
# CALCULATED ACROSS all animals
# -> the median of the cognate responses is 1
cogn_rates = {}
for unit in units:
    cogn_rates[unit] = Df.groupby(['unit','stim']).get_group((unit,cognate_odors[unit])).median()['rate']

for i, rec_group in Df.groupby('recording'):
    for unit in units:
        rec_unit_group = rec_group.groupby('unit').get_group(unit)
        Df.loc[rec_unit_group.index, 'rate_norm_grand'] = rec_unit_group['rate'].values / cogn_rates[unit]

# %%
"""
 
 ########     ###    ########  ########  ##        #######  ########  ######  
 ##     ##   ## ##   ##     ## ##     ## ##       ##     ##    ##    ##    ## 
 ##     ##  ##   ##  ##     ## ##     ## ##       ##     ##    ##    ##       
 ########  ##     ## ########  ########  ##       ##     ##    ##     ######  
 ##     ## ######### ##   ##   ##        ##       ##     ##    ##          ## 
 ##     ## ##     ## ##    ##  ##        ##       ##     ##    ##    ##    ## 
 ########  ##     ## ##     ## ##        ########  #######     ##     ######  
 
"""
# rate_key = 'rate_norm'
# rate_key = 'peak_time'
rate_key = 'onset'

# %% removing
ix = Df.groupby(['stim','unit']).get_group(('A','ab3B')).index
Df.loc[ix,'peak_time'] = np.nan
Df.loc[ix,'onset'] = np.nan

ix = Df.groupby(['stim','unit']).get_group(('B','ab3A')).index
Df.loc[ix,'peak_time'] = np.nan
Df.loc[ix,'onset'] = np.nan

# %% 
order = ['A','A96B','A48B','A24B','A12B','A6B','A3B','AB','B3A','B6A','B12A','B24A','B48A','B96A','B']

fig, axes = plt.subplots(figsize=mm2inch(page_width * 1/3, 60))
sns.boxplot(x='stim', y=rate_key, data=Df, hue='unit', order=order, hue_order=['ab3A','ab3B'], saturation=0.8, linewidth=0.65, fliersize=0.5, ax=axes)
sns.swarmplot(x='stim', y=rate_key, data=Df, hue='unit', order=order, hue_order=['ab3A','ab3B'], dodge=True, size=2, linewidth=0.65, alpha=0.8, ax=axes)

# %% original stats
### STATS
# make room for the significance stars
y1,y2 = axes.get_ylim()
new_y2 = y1 + (y2-y1)*1.1
axes.set_ylim([y1,new_y2])   

yposs = [0.9,0.95]
for i, unit in enumerate(units):
    print(unit)
    a = Df.groupby(['stim','unit']).get_group((cognate_odors[unit],unit))[rate_key].values
    ps = []
    for label in order:
        b = Df.groupby(['stim','unit']).get_group((label, unit))[rate_key].values
        p = stats.mannwhitneyu(a,b)[1]
        # p = stats.ttest_1samp(a-b, 0).pvalue
        print(label, p)
        ps.append(p)

    Ps = pd.Series(ps, index=order)
    F = FDR_correct(Ps, nComp=Ps.shape[0]-1) # no self comparison
    F.loc[unit[-1]] = '' # remove self
    if rate_key == 'peak_time':
        if unit == 'ab3A':
            F.loc['B'] = ''
        if unit == 'ab3B':
            F.loc['A'] = ''

    for label in order:
        xpos = order.index(label)
        ypos = yposs[i]

        # data -> axes coordinate conversion
        xy = axes.transData.transform((xpos,0)) 
        xy = axes.transAxes.inverted().transform(xy)

        axes.annotate(F.loc[label],xy=(xy[0],ypos), xycoords='axes fraction', ha='center',color=unit_colors[unit])

# deco
match rate_key:
    case 'rate':
        descriptor_string = 'Spike rate (1/s)'
    case 'rate_norm':
        descriptor_string = 'Spike rate\nnormalized to cognate odorant' # per animal
    case 'rate_norm_grand':
        descriptor_string = 'Spike rate\nnormalized to cognate odorant'
    case 'peak_time':
        descriptor_string = 'peak time (s)'
    case 'onset':
        descriptor_string = 'onset (ms)'

axes.set_xlabel('')
axes.set_ylabel(descriptor_string)

# thinner box
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(0.4)
axes.xaxis.set_tick_params(width=0.4)
axes.yaxis.set_tick_params(width=0.4)

# rotate ticklabels
axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

# horizontal bars for visual guide
w = 0.20
[axes.axvspan(p-w/2, p+w/2, lw=0, alpha=0.1, color='k', zorder=-1) for p in sp.arange(15)]


sns.despine(fig)
# font size
[tick.set_size('smaller') for tick in axes.get_xticklabels()]

# title string
title_string = 'Response peak rate'
conc = 'dil3'

if conc == 'dil2':
    axes.set_title(title_string + ' at high concentration')
if conc == 'dil3':
    axes.set_title(title_string + ' at low concentration')

axes.set_title('')

# fig sizing
fig.tight_layout()
# fig.subplots_adjust(right=0.78)

# legend
leg_handles, leg_labels = axes.get_legend_handles_labels()
axes.legend(leg_handles[:2], leg_labels[:2], loc='center left', bbox_to_anchor=(1, 0.5))
# axes.legend(leg_handles[:2], leg_labels[:2], loc='center', bbox_to_anchor=(0.5, 0.1))

# save output
if save:
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_summary.pdf")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_summary.svg")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_summary.png", dpi=300)

if rate_key == 'peak_time':
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_peak_time.pdf")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_peak_time.svg")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_peak_time.png", dpi=300)

if rate_key == 'onset_time':
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_onset_time.pdf")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_onset_time.svg")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_onset_time.png", dpi=300)

# %%
"""
 
  ######  ##        #######  ########  ########  ######  
 ##    ## ##       ##     ## ##     ## ##       ##    ## 
 ##       ##       ##     ## ##     ## ##       ##       
  ######  ##       ##     ## ########  ######    ######  
       ## ##       ##     ## ##        ##             ## 
 ##    ## ##       ##     ## ##        ##       ##    ## 
  ######  ########  #######  ##        ########  ######  
 
"""
save = False
unit = 'ab3B'

dts = [np.nan, 96, 48, 24, 12, 6, 3, 0, -3, -6, -12, -24, -48, -96, np.nan]
dts = dict(zip(order, dts))

ddf = []
rate_key = 'rate_norm'
# rate_key = 'rate'
for (rec, label), df in Df.groupby(['recording','stim']):
    if label != 'Air':
        a = df.groupby('unit').get_group('ab3A')[rate_key].values[0]
        b = df.groupby('unit').get_group('ab3B')[rate_key].values[0]
        ddf.append(pd.Series(dict(ab3A=a, ab3B=b, diff=a-b, recording=rec, label=label, dt=dts[label])))

ddf = pd.DataFrame(ddf)

# data retriever helper
def get_data(df, unit):
    x = df['dt'].values
    nan_ix = pd.isna(x)
    y = df[unit].values
    x = x[~nan_ix]
    y = y[~nan_ix]
    return x, y

def fit_lin(x, y):
    res = stats.linregress(x,y)
    m = res.slope
    b = res.intercept
    p = res.pvalue
    return m, b, p

# figure
fig, axes = plt.subplots(figsize=mm2inch(page_width/3,60))

xfit = np.linspace(0, 96, 1000)
recordings = ddf['recording'].unique()
colors = sns.color_palette('husl', n_colors=len(recordings))
colors = dict(zip(recordings,colors))
ps = []
for recording, df in ddf.groupby('recording'):
    # plot each recording
    x, y = get_data(df, unit)
    x = np.absolute(x)
    axes.scatter(x, y, s=8, color=colors[recording], alpha=0.75, linewidth=0, label=recording)
    # stats
    m, b, p = fit_lin(x, y)
    print(recording, p)
    ps.append(p)
    lw = 1 if p < 0.05 else 0.5
    axes.plot(xfit, m*xfit+b, lw=lw, color=colors[recording])

# lin model with all recordings
x, y = get_data(ddf, unit)
x = np.absolute(x)
res = stats.linregress(x,y)
m = res.slope
p = res.intercept
axes.plot(xfit, m*xfit+b, lw=1.2, color='k', zorder=10)
print("all recordings -  slope: %e - p: %e - r2: %.3f" % (m, res.pvalue, res.rvalue**2))

# plot model error
if 0:
    # get lin model errors
    res = stats.linregress(x,y)
    m_err = res.stderr
    b_err = res.intercept_stderr
    N = 1000
    yfits = []
    for i in range(N):
        mp = m + np.random.randn() * m_err
        bp = b + np.random.randn() * b_err
        yfits.append(mp*xfit+bp)

    yfits = np.array(yfits)

    # individual model lines
    # for i in range(N):
    #     axes.plot(xfit, yfits[i], color='k', alpha=0.2)
    
    # shade of 5,95 conf int
    pc = np.percentile(yfits,(5,95),axis=0)
    axes.fill_between(xfit, pc[0], pc[1], color='k', alpha=0.25, linewidth=0, zorder=9)

# deco
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(0.4)
axes.xaxis.set_tick_params(width=0.4)
axes.yaxis.set_tick_params(width=0.4)

axes.set_xlabel('|onset $\Delta$t| (ms)')
axes.set_ylabel('$\Delta$ spike rate (normalized)')
axes.set_xticks([0,3,6,12,24,48,96])
sns.despine(fig, top=True,right=True)
axes.axhline(1, linestyle=':',color='k',lw=0.5)
# axes.set_title(unit)
fig.tight_layout()

# save output
if save:
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_slopes.pdf")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_slopes.svg")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/dts/dts_slopes.png", dpi=300)



# %% using curve fit

# %% 
from scipy.stats.distributions import skewnorm, norm
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 166

def gskew(x, mu, sig, a, s, b):
    y = s * skewnorm(loc=mu, scale=sig, a=a).pdf(x) + b
    return y

def gskew_reduced(x, mu, sig, a, s):
    y = 1 - (s * skewnorm(loc=mu, scale=sig, a=a).pdf(x))
    return y

def gnorm(x, mu, sig, s, b):
    y = s * norm(loc=mu, scale=sig).pdf(x) + b
    return y

def gnorm_reduced(x, mu, sig, s):
    y = 1 - (s * norm(loc=mu, scale=sig).pdf(x))
    return y

# use only signifiant slopes?
sig_recs = recordings[np.array(ps) < 0.05]
df = ddf.loc[[v in sig_recs for v in ddf['recording'].values]]

x, y = get_data(ddf,'ab3B')
x = x *-1
# symmetric?
# x = np.absolute(x) 

# model selection - full thing, not symmetric
model = gskew # mu, sig, a, s, b
lower = np.array([-150,  1e-1, -np.inf, -np.inf, -1000])
upper = np.array([+150, +200,   np.inf, np.inf,  +1000])
p0 = (0, 100, 10, -0.7, 1)

# model selection - reduced, not symmetric
model = gskew_reduced # mu, sig, a, s
lower = np.array([-150,  1e-1, -np.inf, -np.inf])
upper = np.array([+150, +200,   np.inf, np.inf])
p0 = (0, 100, 10, -0.7)

# model selection - reduced, not symmetric
model = gnorm_reduced # mu, sig, s
lower = np.array([-150,  1e-1, -np.inf])
upper = np.array([+150, +200,   np.inf])
p0 = (0, 100, -0.7)


# running the model fits
res = curve_fit(model, x, y, p0=p0, bounds=(lower,upper))
print(res[0])


fig, axes = plt.subplots()
c = sns.color_palette('tab10', n_colors=1)[0]
axes.scatter(x, y, s=8, color='k', alpha=0.45, linewidth=0)

xfit = np.linspace(-150, 150, 100)
yfit = model(xfit, *res[0])
axes.plot(xfit, yfit, color=c)

pfit, pcov = res
perr = np.sqrt(np.diag(pcov))

N = 1000
yfits = []
for i in range(N):
    p = pfit + np.random.randn(pfit.shape[0]) * perr
    y = model(xfit, *p)
    if ~np.any(pd.isna(y)):
        yfits.append(y)
yfits = np.array(yfits)

# shade of 5,95 conf int
pc = np.percentile(yfits,(0.5,99.5),axis=0)
axes.fill_between(xfit, pc[0], pc[1], color=c, alpha=0.25, linewidth=0, zorder=-9)
pc = np.percentile(yfits,(5,95),axis=0)
axes.fill_between(xfit, pc[0], pc[1], color=c, alpha=0.25, linewidth=0, zorder=-8)

sns.despine(fig)
axes.axhline(1, linestyle=':',color='k',lw=0.5)
axes.set_xlabel("Time (ms)")
axes.set_ylabel('$\Delta$ spike rate (normalized)')
