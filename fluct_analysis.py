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

zoom = True
save = False

pre, post = (-1,12) * pq.s

# high concentration example
config_path = Path("/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/161202_C/dil3/fluct_dil3/sssort_config_fluct_dil3.ini")

# low concentration example
# config_path = Path("/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/161202_C/dil2/fluct_dil2/sssort_config_fluct_dil2.ini")

# get ephys data
blk = get_blk(config_path, pre, post)
trial_labels = [seg.annotations['label'] for seg in blk.segments]

# PID read
import configparser
Config = configparser.ConfigParser()
Config.read(config_path)
data_path = Path(Config.get('path', 'data_path'))
letter = data_path.stem.split('_')[2]

pid_path = "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/PID/PID_fluct_%s.dill" % letter.upper()
with open(pid_path,'rb') as fH:
    pid_blk = dill.load(fH)

# definition of mapping
PID_map = {'ABa':['A','A'],
           'ABb':['B','B'],
           'ABi':['A','B'],
           'A'  :['A',''],
           'B'  :['','B'],}

pid_traces_norm = {}
for label in np.unique(trial_labels):
    segs = select_by_dict(pid_blk.segments, label=label)
    data = []
    for seg in segs:
        asig = seg.analogsignals[0] # the PID channel
        data.append(asig.magnitude.flatten())
    data = np.array(data)
    data = np.average(data, axis=0)

    # hardcoded baseline removal 
    data = data - np.average(data[:12000]) # 1/2 second
    data = data / data.max()

    # hardcoded mild filtering
    data = np.convolve(data, np.ones(25)/25, mode='same')
    pid_traces_norm[label] = data

pid_traces_norm[''] = np.zeros(data.shape[0])

# get rio sequence
sqc_path = "/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/RIO_procotols/2016_11_28_HEPN_MHXE_fluct_tau50_%s/HEPN_MHXE_fluct_tau50_templates_corr.sqc" % letter.lower()
rio_seq = rio.read_sqc(sqc_path)


# %% inspect plots
inspect_labels = ['ABi']

for label in inspect_labels:

    fig, axes = plt.subplots(nrows=3,
                            sharex=True,
                            gridspec_kw=dict(height_ratios=[1,1,4]),
                            figsize=(mm2inch(page_width/2, page_width/2)))
    
    # get data
    segs = select_by_dict(blk.segments,label=label)
    tvec = segs[0].analogsignals[0].times - segs[0].analogsignals[0].times[0] + pre
    
    # plot stimulus
    ix = [Pattern.name for Pattern in rio_seq.Patterns].index(label +'_tau50ms')
    Pattern = rio_seq.Patterns[ix]

    ch_flip = {0:1,1:0} # channel flip

    for i in range(2):
        j = ch_flip[i]
        ch = Pattern.calc_states(j).astype('int32')
        n_pre = (np.absolute(pre).magnitude * 1000).astype('int32')
        n_post = (np.absolute(post - 0.5*pq.s).magnitude * 1000).astype('int32')
        ch = np.pad(ch,(n_pre, n_post), mode='constant', constant_values=0)

        tvec_p = np.linspace(tvec[0], tvec[-1], ch.shape[0])
        axes[0].plot(tvec_p, ch*0.8 + i*1,
                     color=pid_colors[odors[i]],
                     alpha=0.8, lw=0.5, label=odors[i])

    axes[0].legend(loc='upper right', fontsize='smaller')
    axes[0].set_ylim([-0.2,2])
    axes[0].set_ylabel('Valve state')
    axes[0].yaxis.set_major_locator(plt.NullLocator())
    
    # pid traces
    tvec = segs[0].analogsignals[0].times - segs[0].analogsignals[0].times[0] + pre
    for i, l in enumerate(PID_map[label]):
        pid_trace = pid_traces_norm[l]
        axes[1].plot(tvec, pid_trace, color=pid_colors[odors[i]],
                     lw=0.5, alpha=0.8, label=odors[i])

    axes[1].set_ylabel('PID')
    axes[1].set_ylim(-0.1,1.2)
    plt.setp(axes[1].get_xticklabels(), visible=False)
    
    ### plot data: traces with predicted spikes
    # traces with predicted spikes
    ymax = blk.segments[0].analogsignals[0].max()
    yscl = 1/ymax * 0.75

    for i, seg in enumerate(segs):
        # the recording
        asig = seg.analogsignals[0]
        tvec = asig.times - asig.times[0] + pre
        axes[2].plot(tvec, asig * yscl + i, lw=0.4, color='k', alpha=0.75)

        # the spiketrains
        for unit in units:
            st, = select_by_dict(seg.spiketrains, unit=unit)
            for t in st.times:
                t_start, t_stop = (t-2*pq.ms, t+2*pq.ms)
                if t_start > asig.t_start and t_stop < asig.t_stop:
                    spike_waveform = asig.time_slice(t-2*pq.ms,t+2*pq.ms)
                    spike_times = spike_waveform.times-asig.times[0] + pre
                    axes[2].plot(spike_times, spike_waveform * yscl + i,
                                 color=unit_colors[unit], lw=0.45, label=unit)
   
    ### deco
    plt.setp(axes[0].get_xticklabels(), visible=False)
    axes[2].legend(loc='upper right')
    axes[2].yaxis.set_major_locator(plt.NullLocator())
    axes[2].set_ylabel('Recorded spikes')
    axes[2].set_xlabel('Time (s)')
    
    # removes duplicates
    # custom legend
    handles, labels = axes[2].get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(list(zip(labels, handles)))
    axes[2].legend(list(by_label.values()), list(by_label.keys()),loc='upper right',fontsize='smaller')
    

    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    
    conc = config_path.stem.split('_')[-1]
    if conc == 'dil3':
        title = "Low concentration"
    if conc == 'dil2':
        title = "High concentration"
    fig.suptitle(title)

    for ax in axes:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.4)
        ax.xaxis.set_tick_params(width=0.4)
        ax.yaxis.set_tick_params(width=0.4)
    
    # if zoom:
    if zoom:
        axes[2].set_xlim(4,6)
    else: 
        axes[2].set_xlim(-0.5,10.5)
    
    # save output
    if save:
        fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/fluct/fluct_example_%s_%s.pdf" % (label, conc))
        fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/fluct/fluct_example_%s_%s.svg" % (label, conc))
        fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/fluct/fluct_example_%s_%s.png" % (label, conc), dpi=300)


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
# just reading in
pre, post = (-1,12) * pq.s

# both concentrations
config_paths = Path("/media/georg/data/SSR_thesis_data/SSR_data/3_reuse/all_fluct_results")
config_paths = np.genfromtxt(config_paths,dtype='U')
concs = ['dil'+path[-5] for path in config_paths]

blks = []
for config_path in config_paths:
    config_path = Path(config_path)
    print(config_path)
    blk = get_blk(config_path, pre, post)
    blks.append(blk)

# %% victor purpura dists
from itertools import permutations,combinations
nReps = 5
# self combos
inner_combos = list(combinations(range(nReps),2))
# across combos
across_combos = list(permutations(range(nReps),2))
[across_combos.append((r,r)) for r in range(nReps)]

comparisons = {'ab3A':(('ABi','ABi'),('A','A'),('ABi','A'),('ABa','A')),
               'ab3B':(('ABi','ABi'),('B','B'),('ABi','B'),('ABb','B'))}

Df = []
for unit in comparisons.keys():
    for comp in comparisons[unit]:
        for blk_ix, blk in enumerate(blks):
            # animal_name = os.path.splitext(os.path.basename(blk.annotations['data_path']))[0]
            animal_name = Path(config_paths[blk_ix]).parts[-4]

            segs = select_by_dict(blk.segments,label=comp[0])
            sts_a = []
            for seg in segs:
                st, = select_by_dict(seg.spiketrains, unit=unit)
                st_shifted = st - seg.annotations['stim_time']
                sts_a.append(st_shifted)

            segs = select_by_dict(blk.segments,label=comp[1])
            sts_b = []
            for seg in segs:
                st, = select_by_dict(seg.spiketrains, unit=unit)
                st_shifted = st - seg.annotations['stim_time']
                sts_b.append(st_shifted)

            if comp[0] == comp[1]:
                combos = inner_combos
            else:
                combos = across_combos
                
            for c, (i,j) in enumerate(combos):
                Dvp = ele.spike_train_dissimilarity.victor_purpura_dist([sts_a[i], sts_b[j]])[0,1]
                S = pd.Series(dict(animal=animal_name, conc=concs[blk_ix], comp='-'.join(comp), unit=unit, vpdist=Dvp))
                Df.append(S)

Df = pd.DataFrame(Df)

# %%
save = False

fig, axes = plt.subplots(ncols=5, sharey=False, figsize=mm2inch(page_width* 3/4,60), gridspec_kw=dict(width_ratios=(1,1,.3,1,1)))
axes[2].remove()
colors = sns.color_palette('muted',n_colors=4)

kwargs = dict(x='comp', y='vpdist', linewidth=0.65, fliersize=0.5)

axix = {('ab3A','dil3'):0,
        ('ab3B','dil3'):1,
        ('ab3A','dil2'):3,
        ('ab3B','dil2'):4}

for i, (group, df) in enumerate(Df.groupby(['unit','conc'])):
    unit, conc = group
    sns.boxplot(data=df, ax=axes[axix[group]], palette=[unit_colors[unit]], **kwargs)

axes[0].set_ylim(axes[1].get_ylim())
axes[1].set_yticks([])
axes[3].set_ylim(axes[4].get_ylim())
axes[4].set_yticks([])

ab3A_labels = ['$(AB_{i},AB_{i})$','$(A,A)$','$(AB_{i},A)$','$(AB_{A},A)$']
ab3B_labels = ['$(AB_{i},AB_{i})$','$(B,B)$','$(AB_{i},B)$','$(AB_{B},B)$']

for i in [0,1,3,4]:
    axes[i].tick_params(axis='x', pad=0)

axes[0].set_xticks(range(4), ab3A_labels, rotation=45, ha='right', va='top')
axes[1].set_xticks(range(4), ab3B_labels, rotation=45, ha='right', va='top')

axes[3].set_xticks(range(4), ab3A_labels, rotation=45, ha='right')
axes[4].set_xticks(range(4), ab3B_labels, rotation=45, ha='right')

axes[0].set_title('ab3A')
axes[1].set_title('ab3B')

axes[3].set_title('ab3A')
axes[4].set_title('ab3B')

axes[0].set_ylabel('$D_{VP}$')
for ax in axes[1:]:
    ax.set_ylabel('')

for ax in axes:
    ax.set_xlabel('')

def axes2figure(xy, axes, fig):
    xy = axes.transAxes.transform(xy)
    xy_t = fig.transFigure.inverted().transform(xy)
    return xy_t

x = np.average([axes2figure((0.5,1), axes[i], fig)[0] for i in [0,1]])
fig.text(x, 0.95, 'Low concentration',ha='center', fontsize=plotting_globals.MEDIUM_SIZE)

x = np.average([axes2figure((0.5,1), axes[i], fig)[0] for i in [3,4]])
fig.text(x, 0.95, 'High concentration',ha='center', fontsize=plotting_globals.MEDIUM_SIZE)

# fig.tight_layout()
fig.subplots_adjust(top=0.85, bottom=0.15)

for i in [0,1,3,4]:
    ax = axes[i]
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.4)
    ax.xaxis.set_tick_params(width=0.4)
    ax.yaxis.set_tick_params(width=0.4)

sns.despine(fig)

### STATS

# helpers
def axes2data(xy, axes):
    xy = axes.transAxes.transform(xy)
    xy_t = axes.transData.inverted().transform(xy)
    return xy_t

def get_stats(conc, unit, comp_a, comp_b, Df):
    # if 1: # this is reading in Giovannis stats
        # df = pd.read_csv()
    
    # this is computing it with pingouin by hand
    # import pingouin as pg
    # values_a = Df.groupby(['conc','comp']).get_group((conc, comp_a))['vpdist'].values
    # values_b = Df.groupby(['conc','comp']).get_group((conc, comp_b))['vpdist'].values
    # p = pg.ttest(values_a, values_b)['p-val'][0]

    # if 0: # this is the old way
    values_a = Df.groupby(['conc','comp']).get_group((conc, comp_a))['vpdist'].values
    values_b = Df.groupby(['conc','comp']).get_group((conc, comp_b))['vpdist'].values
    p = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')[1]
    print("conc: %s, unit: %s, stim: %s vs %s:, p=%.4f = %e" % (conc, unit, comp_a, comp_b, p*3, p*3))
    return p

def annotate_stats(p, ax, xps, yp):
    x, y = axes2data((0, yp), ax)
    ax.annotate(p2stars(p, ns=True) ,xy=(np.average(xps), y), xycoords='data', ha='center',fontsize='small',color='black')
    x, y = axes2data((0, yp-0.01), ax)
    ax.plot(xps, [y, y] ,lw=0.5, color='k')

# make room for the significance stars
for ax in axes[:2]:
    y1,y2 = ax.get_ylim()
    new_y2 = y1 + (y2-y1) * 1.1
    ax.set_ylim([y1,new_y2])

for ax in axes[3:]:
    y1,y2 = ax.get_ylim()
    new_y2 = y1 + (y2-y1) * 1.1
    ax.set_ylim([y1,new_y2])

## comparisons
    
# first
conc = 'dil3'
unit = 'ab3A'
ax = axes[0]
p = get_stats(conc, unit, 'ABi-ABi','A-A',Df) * 3
annotate_stats(p, ax, (0,1), 0.8)

p = get_stats(conc, unit, 'A-A','ABi-A',Df) * 3
annotate_stats(p, ax, (1,2), 0.85)

p = get_stats(conc, unit, 'ABi-A','ABa-A',Df) * 3
annotate_stats(p, ax, (2,3), 0.9)

# second
conc = 'dil3'
unit = 'ab3B'
ax = axes[1]
p = get_stats(conc, unit, 'ABi-ABi','B-B',Df) * 3
annotate_stats(p, ax, (0,1), 0.8)

p = get_stats(conc, unit, 'B-B','ABi-B',Df) * 3
annotate_stats(p, ax, (1,2), 0.85)

p = get_stats(conc, unit, 'ABi-B','ABb-B',Df) * 3
annotate_stats(p, ax, (2,3), 0.9)

# third
conc = 'dil2'
unit = 'ab3A'
ax = axes[3]
p = get_stats(conc, unit, 'ABi-ABi','A-A',Df) * 3
annotate_stats(p, ax, (0,1), 0.8)

p = get_stats(conc, unit, 'A-A','ABi-A',Df) * 3
annotate_stats(p, ax, (1,2), 0.85)

p = get_stats(conc, unit, 'ABi-A','ABa-A',Df) * 3
annotate_stats(p, ax, (2,3), 0.9)

# forth
conc = 'dil2'
unit = 'ab3B'
ax = axes[4]
p = get_stats(conc, unit, 'ABi-ABi','B-B',Df) * 3
annotate_stats(p, ax, (0,1), 0.8)

p = get_stats(conc, unit, 'B-B','ABi-B',Df) * 3
p *= 2 # bonferroni correction
annotate_stats(p, ax, (1,2), 0.85)

p = get_stats(conc, unit, 'ABi-B','ABb-B',Df) * 3
annotate_stats(p, ax, (2,3), 0.9)

# save output
if save:
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/fluct/fluct_summary.pdf")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/fluct/fluct_summary.svg")
    fig.savefig("/home/georg/Projects/SSR_paper_paul/figures/fluct/fluct_summary.png", dpi=300)


"""
conc: dil3, unit: ab3A, stim: ABi-ABi vs A-A:, p=0.3575
conc: dil3, unit: ab3A, stim: A-A vs ABi-A:, p=0.3138
conc: dil3, unit: ab3A, stim: ABi-A vs ABa-A:, p=0.3473
conc: dil3, unit: ab3B, stim: ABi-ABi vs B-B:, p=0.3753
conc: dil3, unit: ab3B, stim: B-B vs ABi-B:, p=0.0063
conc: dil3, unit: ab3B, stim: ABi-B vs ABb-B:, p=0.0000
conc: dil2, unit: ab3A, stim: ABi-ABi vs A-A:, p=0.7673
conc: dil2, unit: ab3A, stim: A-A vs ABi-A:, p=0.0021
conc: dil2, unit: ab3A, stim: ABi-A vs ABa-A:, p=0.1620
conc: dil2, unit: ab3B, stim: ABi-ABi vs B-B:, p=0.2421
conc: dil2, unit: ab3B, stim: B-B vs ABi-B:, p=0.0000
conc: dil2, unit: ab3B, stim: ABi-B vs ABb-B:, p=0.0000
"""


# %% giovanni
"""
Unterscheiden sich [(ABi,ABi), (A,A) gepoolt] von [(ABi,A),(ABa,A) gepoolt] -> es gibt ephaptik.
Unterscheiden sich (ABi,A) von (ABa,A) -> ephaptic ist synchron stärker.
"""
a = pd.concat([Df.groupby(['comp','unit']).get_group(('ABi-ABi','ab3A')),
               Df.groupby(['comp','unit']).get_group(('A-A','ab3A'))])['vpdist'].values

b = pd.concat([Df.groupby(['comp','unit']).get_group(('ABi-A','ab3A')),
               Df.groupby(['comp','unit']).get_group(('ABa-A','ab3A'))])['vpdist'].values

p = stats.mannwhitneyu(a,b,alternative='two-sided')[1]
print(p)

a = pd.concat([Df.groupby(['comp','unit']).get_group(('ABi-ABi','ab3B')),
               Df.groupby(['comp','unit']).get_group(('B-B','ab3B'))])['vpdist'].values

b = pd.concat([Df.groupby(['comp','unit']).get_group(('ABi-B','ab3B')),
               Df.groupby(['comp','unit']).get_group(('ABb-B','ab3B'))])['vpdist'].values

p = stats.mannwhitneyu(a,b,alternative='two-sided')[1]
print(p)
