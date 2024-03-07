# -*- coding: utf-8 -*-
"""
This library can be used to programmatically read, create, manipulate .sqc files
as used by the cRIO pulse interface developed by Stefanie Neupert
(stefanie.neupert@uni-konstanz.de) in the lab of Christoph Kleineidam at the 
University of Konstanz.

The internal logic of the RIO system, that each pulse is part of a pattern, and 
a sequence consists of multiple patterns is reflected in the OO structure of this
library.

RIOhttps://github.com/grg2rsr/RIOlib

@author: 
---------------------------------------
Georg Raiser, PhD Candidate

georg.raiser@uni-konstanz.de
grg2rsr@gmail.com

Dept. of Neurobiology, Prof. CG Galizia
University of Konstanz, Germany
Phone: +49-7531-88-2102
---------------------------------------

"""

import sys
import os

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


#==============================================================================
# Classes for OO representation
#==============================================================================

class RIOsequence(object):
    """ The OO representation of a RIO sequence. Contains a list of its patterns,
    as well as the delay between the patterns.
    """
    def __init__(self, seq_name='', Patterns=None, delay_btw_patterns=0):
        """        
        Attributes
        ----------
        seq_name : str
            the name of the sequence
            
        Patterns : list[RIOpattern]
            a list of RIOpatterns. If not provided, and empty sequence is
            instantiated
            
        delay_btw_patterns : int
            the delay between patterns (in ms)
        """
        
        self.seq_name = seq_name
        self._Header = []
        self._Lines = []
        self._total_length = 0
        self._delay_btw_patterns = delay_btw_patterns
        
        self.Patterns = []
        if Patterns:
            self.add_Patterns(Patterns)
        self._update()
        
    def add_Pattern(self,Pattern):
        """ Adds a RIOpattern to the sequence

        Parameters
        ----------
        Pattern: RIOpattern
            The RIOpattern to append as the last pattern

        """
        self.Patterns.append(Pattern)
        self._update()
        pass
    
    def add_Patterns(self,Patterns):
        """ Convenience function to add a list containing RIOpatterns to the 
        sequence

        Parameters
        ----------
        Patterns: list[RIOpattern]
            The RIOpattern to append as the last pattern
        """
        for Pattern in Patterns:
            self.add_Pattern(Pattern)
        self._update()
        
    def preview_plot(self):
        """ graphically plots the entire sequence for visual inspection

        Note
        ----
        this opens a matplotlib window that can contain many points, depending
        on the lenght of the sequence. This might take a while.

        """
        ### FIXME 2do!
        pass
        
    def write_sqc(self,path):
        """ writes the sequence to an .sqc file.

        Parameters
        ----------
        outpath: str
            the path to write the .sqc file to.

        """
        
        self._update()
        
        if os.path.splitext(path)[1] == '':
            # a directory is passed, filename is then seq name
            path = os.path.join(path,self.seq_name) + '.sqc'
            
        # make directory if it doesn't exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        with open(path,'w') as fh:
            for line in self._Header + self._Lines:
                fh.write(line)
                
        print("sequence with name "  + self.seq_name + " written to " + path)

    def _update(self):
        """ private method: internal updater """
        self._total_length = sum([Pattern.total_duration for Pattern in self.Patterns]) + self.delay_btw_patterns * (len(self.Patterns) - 1)
        self._generate_Header()
        self._generate_Lines()
        
    def get_delay_btw_patterns(self):
        """ getter for delay_btw_patterns """
        return self._delay_btw_patterns
        
    def set_delay_btw_patterns(self,val):
        """ setter for delay_btw_patterns """
        self._delay_btw_patterns = val
        self._update()
    
    delay_btw_patterns = property(get_delay_btw_patterns,set_delay_btw_patterns)
        
    def _generate_Lines(self):
        """ private method: generating the lines for writing the .sqc """
        self._Lines = []
        for i,Pattern in enumerate(self.Patterns):
            Pattern._generate_Lines()
            for line in Pattern._Lines:
                self._Lines.append(line)
                
    def _generate_Header(self):
        """ private method: generating the .sqc header """
        self._Header = []
        self._Header.append("Sequence name:\t" + self.seq_name + '\r\n')
        self._Header.append("Total sequence dur. (ms):\t" + str(self._total_length) + '\r\n')
        self._Header.append("Delay between patterns (ms):\t" + str(self.delay_btw_patterns) + '\r\n')
        self._Header.append('\r\n')
        self._Header.append('\t'.join(['ch','# of pulses','start (ms)','Duration (ms)','ISI (ms)','label','total duration (ms)','flag (next pattern)','Pattern Name','Concentration'+'\r\n']))
        
        
class RIOpattern(object):
    """ The OO representation of a RIO pattern. Contains a list of its pulses,
    as well as the total duration of the pattern
    """
    
    def __init__(self, name='', total_duration=None, Pulses=None):
        """        
        Attributes
        ----------
        name : str
            the name of the pattern
            
        total_duration: int
            the length of the pattern. If ommited, it is automatically calculated
            to the minimum length
        
        Pulses : list[RIOpulses]
            a list of RIOpulses. If ommited, and empty pattern is instantiated
        """
        self.name = name
               
        if total_duration == None:
            self.total_duration = 0
            self.duration_defined = False
        else:
            self.total_duration = total_duration
            self.duration_defined = True
        
        self.Pulses = []
        if Pulses:
            self.add_pulses(Pulses)
            
        self._Lines = []
        self._update()
    
    def add_pulse(self,Pulse):
        """ Adds a RIOpulse to the pattern

        Parameters
        ----------
        Pulse: RIOpulse
            The RIOpulse to add to this pattern.

        """
        self.Pulses.append(Pulse)
        self._update()
        
    def add_pulses(self,Pulses):
        """ Convenience function for adding a list of RIOpulses to the pattern.

        Parameters
        ----------
        Pulses: list[RIOpulse]
            The list containint the RIOpulses to add to this pattern.

        """
        for Pulse in Pulses:
            self.add_pulse(Pulse)
        self._update()

    def add_pattern(self,Pattern):
        """ Adds an entire pattern to this pattern.

        Note
        ----
        Be aware that can cause conflicting information for individual channels.

        Parameters
        ----------
        Pattern: RIOpattern
            The Pattern to add to this pattern.

        """
        for Pulse in Pattern.Pulses:
            self.add_pulse(Pulse)
        self._update()
        
    def calc_states(self,channel):
        """Calculates the state vector for a single channel

        Parameters
        ----------
        channel: int
            the channel to calculate the states from
        
        Returns
        -------
        state_vec: np.array
            the state vector

        """
        state_vec = np.zeros(self.total_duration,dtype='bool')
        channel = str(channel)
        for pulse in self.Pulses:
            if pulse.specs['channel'] == channel:
                start = int(pulse.specs['start'])
                dur = int(pulse.specs['duration'])
                stop = start + dur 
                state_vec[start:stop] = 1
            
        return state_vec

    def preview_plot(self):
        """ graphically plots the pattern for visual inspection

        Note
        ----
        this opens a matplotlib window that can contain many points, depending
        on the lenght of the pattern. This might take a while.
        """
        channels = np.unique([pulse.specs['channel'] for pulse in self.Pulses])
        for i,channel in enumerate(channels):
            state_vec = self.calc_states(channel)
            plt.plot(state_vec*0.8 + int(channel))
            plt.text(100,int(channel)+0.4,int(channel),verticalalignment='center',horizontalalignment='left')
            
        ax = plt.gca()
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('channel')
        ax.set_yticklabels([])
        ax.set_title(self.name)
        ax.margins(0.1,0.1)
        pass
        
    def _update(self):
        """ private method: internal updater """
        
        if self.duration_defined == False: # if not set explicitly, recalculate
            longest_pulse = 0
            for Pulse in self.Pulses:
                Pulse_length = int(Pulse.specs['start']) + int(Pulse.specs['duration'])
                if Pulse_length > longest_pulse:
                    longest_pulse = Pulse_length
            self.total_duration = longest_pulse
            
        self._generate_Lines()
    
    def _generate_Lines(self):
        """ private method: generating the lines for the pattern. This function
        contains the RIO specification """
        self._Lines = []
        for i,Pulse in enumerate(self.Pulses):
            if i == 0:
                # first pulse in pattern gets the name and a '1' for next_pattern_flag as well as the patterns total duration
                Pattern_name = self.name
                flag_next_pattern = '1'
                total_duration = str(self.total_duration)
            else:
                Pattern_name = ''
                flag_next_pattern = ''
                total_duration = ''
    
            self._Lines.append('\t'.join([Pulse.specs['channel'],Pulse.specs['number'],Pulse.specs['start'],Pulse.specs['duration'],Pulse.specs['ISI'],Pulse.specs['label'],total_duration,flag_next_pattern,Pattern_name,Pulse.specs['concentration'],'\r\n']))
        pass

        
class RIOpulse(object):
    """ The OO representation of a single RIO pulse. Not really necessary since 
    a pulse is essentially a dict, but kept for consistency and possible later 
    extension.""" 
    def __init__(self,channel,number,start,duration,ISI='',label='',concentration=''):
        """        
        Attributes
        ----------
        channel
            the channel to switch
            
        number
            the number of pulses
        
        start
            the start of the pulse (in ms)
        
        duration
            the duration of the pulse (in ms)
            
        ISI
            the inter-stimulus-interval (in ms), defined as 'start to start'
        
        label
            the label of the pulse
            
        concentration
            the concentration of the stimulus (quite odor specific, never used 
            in this library)
        """
      
        self.specs = {'channel': str(channel),
                      'number': str(number),
                      'start': str(start),
                      'duration': str(duration),
                      'ISI': str(ISI),
                      'label': str(label),
                      'concentration': str(concentration)}
        pass
    


#==============================================================================
# lib methods
#==============================================================================

def read_sqc(sqc_path):
    """Reads an .sqc file into the OO structure of this library.

    Parameters
    ----------
    sqc_path : str
        the path to the .sqc file to read

    Returns
    -------
    Seq: RIOsequence
    
    """
    
    # parse header
    with open(sqc_path, 'r') as fh:
        lines = fh.readlines()
    
    info = [line.split('\t') for line in lines[:3]]
    
    try:
        name = info[0][1].strip()
    except:
        name = ''
        
    total_length = int(info[1][1])
    delay_btw_patterns = int(info[2][1])
    
    # generate OO structure
    Seq = RIOsequence(seq_name=name, delay_btw_patterns=delay_btw_patterns)
    Seq.total_length = total_length
    
    # get data
    Data = pd.read_csv(sqc_path,delimiter='\t',skiprows=4,index_col=False)
    
    # split into Patterns
    pattern_breaks = np.where(Data['flag (next pattern)'] == 1)[0] # new patterns start indices
    pattern_breaks = np.concatenate((pattern_breaks,[Data.shape[0]])) # for indexing last pattern
    
    # loop over patterns
    for i in range(len(pattern_breaks)-1):
        # get Pattern_data
        Pattern_Data = Data.iloc[pattern_breaks[i]:pattern_breaks[i+1]]
        Pattern_Data = Pattern_Data.fillna(value='')
        
        # extract pulse info
        RIOpulses = []
        for j in range(Pattern_Data.shape[0]):
            P = RIOpulse(Pattern_Data.iloc[j]['ch'],
                         Pattern_Data.iloc[j]['# of pulses'],
                         Pattern_Data.iloc[j]['start (ms)'],
                         Pattern_Data.iloc[j]['Duration (ms)'],
                         Pattern_Data.iloc[j]['ISI (ms)'],
                         Pattern_Data.iloc[j]['label'],
                         Pattern_Data.iloc[j]['Concentration'])
            RIOpulses.append(P)
            
        Seq.add_Pattern(RIOpattern(name=Pattern_Data.iloc[0]['Pattern Name'],total_duration=int(Pattern_Data.iloc[0]['total duration (ms)']),Pulses=RIOpulses))
    return Seq


def state_vec2change_times(state_vec):
    """Converts a state vector to a list of change times. A change time is  the
    last index before the change from low to high or vice versa.

    Parameters
    ----------
    state_vec : np.array
        the state vector

    Returns
    -------
    change_times: np.array
        1D, holding the change times
    
    
    """
    change_times = np.where(np.absolute(np.diff(state_vec)) == 1)[0]
    return change_times


def change_times2state_vec(change_times,total_time,initial_state='low'):
    """Converts change times to a state vector. A state vector is a np.array of
    bool or int type, 1 or True denotes 'high' and 0 or False denotes 'low'. Each
    index corresponds to one step of the RIO timebase (which is 1 ms at the 
    moment)

    Parameters
    ----------
    change_times : np.array

    total_time: int
        the total lenght
    
    initial_state: 'low' or 'high'
        denotes whether the first state is low or high. 
    
    Returns
    -------
    state_vec: np.array
        the state vector
    """
    
    state_vec = np.zeros(total_time)
    
    # cast if not array
    if type(change_times) != np.ndarray:
        change_times = np.array(change_times)
    
    # deconstruct, always pairs of two if even, thats all, if uneven, add one at end
    if len(change_times) % 2 == 1:
        # append the last timepoint if last pulse doesn't end
        change_times = np.concatenate((change_times,[total_time]))
        
    # deconstruct into pulses
    for i in range(0,len(change_times),2):
        state_vec[change_times[i]+1:change_times[i+1]+1] = 1
    
    state_vec = state_vec.astype('bool')
    if initial_state == 'high':
        state_vec = np.logical_not(state_vec)
        
    return state_vec


def States2RIOpulses(state_vec,channel,label='',concentration=''):
    """Converts a state vector to a list of RIOpulses

    Parameters
    ----------
    state_vec: np.array
        the state vector

    channel: int
        the channel to contain the pulses
    
    label: str
        optional label for the pulses
        
    concentration: str
        optional concentration for the pulses. Not used in this library.
    
    Returns
    -------
    pulses: list[RIOpulses]
        a list of RIOpulses
    """
    
    total_time = len(state_vec)
    
    # deduce start and final states
    if state_vec[0] == 1:
        initial_state = 'high'
    else:
        initial_state = 'low'
    
    if state_vec[-1] == 1:
        last_state = 'high'
    else:
        last_state = 'low'

    # to change times
    change_times = np.where(np.absolute(np.diff(state_vec)) == 1)[0] + 1 
    
    # add additional
    change_times = np.concatenate((change_times,[total_time]))
    
    if initial_state == 'high':
        change_times = np.concatenate(([0],change_times))
        
    if last_state == 'high':
        change_times = np.concatenate((change_times,[total_time + 1]))
    
    if last_state == 'high': 
        nPulses = (len(change_times)) / 2
    else:
        nPulses = (len(change_times) - 1) / 2
    
    # deconstruct into pulses
    pulses = []
    for i in range(nPulses):
        start = change_times[2*i]
        stop  = change_times[2*i+1]
        nex   = change_times[2*i+2]
        
        pulses.append(RIOpulse(channel,1,start,stop-start,nex-start,label,concentration))

    return pulses
   

#==============================================================================
# helpful lib methods
#==============================================================================

def randomize_Patterns(RIOpatterns, nReps=1, seq_name='', pseudorandom=True, delay_btw_patterns=0):
    """Randomizes patterns and creates a new sequence of them. Takes each pattern
    present in RIOpatterns and repeats them nReps times.

    Parameters
    ----------
    RIOpatterns: list[Riopattern]
        A list of RIOpatterns to take the templates from

    nReps: int
        the number how many times each pattern will be present in the final
        sequence
    
    pseudorandom: bool
        if True, the final sequence will consist of blocks of the randomized
        pattern order in a way that each patterns will presented at least once
        before the first (second, third ...) repetition.
        I.e. if ['A','B','C'] are randomized two times, the result will be two
        random blocks of ['A','B','C'] concatenated, so that 
        ['A','B','A','B','C','C'] is impossible to get by chance.
        
    Returns
    -------
    Seq: RIOsequence
        The new sequence with the randomized order.

    rand_names: list[str]
        A list with the stimulus names, in the randomized order
        
    rand_inds: list[int[]
        A list with the stimulus indices referring to the original template, 
        in the randomized order
        
    """
    names = [RIOpattern.name for RIOpattern in RIOpatterns]
    
    if pseudorandom:
        rand_inds = np.concatenate([np.random.permutation(len(RIOpatterns)) for i in range(nReps)])
    
    else:
        rand_inds = np.random.permutation(list(range(len(RIOpatterns))) * nReps)
    
    rand_names = [names[i] for i in rand_inds]
    Seq = RIOsequence(seq_name=seq_name, Patterns=[RIOpatterns[i] for i in rand_inds], delay_btw_patterns=delay_btw_patterns)
    
    return Seq, rand_names, rand_inds


def compose_Patterns(RIOpatterns,comp_pattern_name,total_duration=None):
    """ merges all Patterns in RIOpatterns (list) to a single pattern """
    composed_Pattern = RIOpattern(name=comp_pattern_name, total_duration=total_duration)
    for Pattern in RIOpatterns:
        composed_Pattern.add_pattern(Pattern)
    
    return composed_Pattern
    
    
#==============================================================================
# some pattern generators
#==============================================================================

def calc_random_pattern_exponential(tau,tStart=0,tDuration=1000,tTotal=1000,channel=0,name=''):
    """Calculate a sequence of random state changes with intervals drawn from an
    exponential distribution.

    Parameters
    ----------
    tau: float
        the scale parameter of the exponential distribution (in ms)

    tStart: int
        the time point at which the random changes start (in ms)
        
    tDuration: int
        the total length of the time section in which random changes can occur
        (in ms)
        
    tTotal: int
        the total length of the pattern (in ms)
        
    channel: int
        the channel to switch
        
    name: str
        the name of the pattern
        
    Returns
    -------
    Pattern: RIOpattern
        The generated RIOpattern instance
        
    """
    
    # a little ugly but guarantees to run
    change_times = [0]
    while np.cumsum(change_times)[-1] < tDuration:
        change_times.append(stats.distributions.expon.rvs(scale=tau))
    
    change_times = np.cumsum(change_times[1:-1]).astype('int32') + tStart
    
    # to make sure it ends latest at tStart+tDuration
    if len(change_times) % 2 == 1:
        change_times = np.concatenate((change_times,[tStart+tDuration]))
    
    state_vec = change_times2state_vec(change_times,tTotal)
    Pattern = RIOpattern(name=name, Pulses=States2RIOpulses(state_vec,channel), total_duration=tTotal)
    return Pattern, state_vec


def calc_random_pattern_blocks(tCorr=100,prob=0.5,tStart=0,tDuration=1000,tTotal=1000,channel=0,name=''):
    """Calculate a sequence of random states with a fixed time length, and a 
    settable probability to be in either state.

    Parameters
    ----------
    tCorr: float
        the length of a state (in ms). Note that this can be a float, but will 
        be ultimately rounded to ms.
    
    prob: float
        the probability for each block to be in the 'high' state.

    tStart: int
        the time point at which the random changes start (in ms)
        
    tDuration: int
        the total length of the time section in which random changes can occur
        (in ms)
        
    tTotal: int
        the total length of the pattern (in ms)
        
    channel: int
        the channel to switch
        
    name: str
        the name of the pattern
        
    Returns
    -------
    Pattern: RIOpattern
        The generated RIOpattern instance    
    
    """
    
    nBlocks = int(tDuration / tCorr)
    Pattern = (np.rand(nBlocks) < prob).astype('float32')
    state_vec = np.repeat(Pattern,tCorr)
    
    # acount for rounding errors
    if state_vec.shape[0] < tDuration:
        state_vec = np.pad(state_vec,(0,tDuration - state_vec.shape[0]), mode='minimum')
    else:
        Pattern = Pattern[:tTotal]
    
    # pad to final size
    state_vec = np.pad(state_vec, (tStart,tTotal-tStart-tDuration), mode='minimum')
    
    Pattern = RIOpattern(name=name, Pulses=States2RIOpulses(state_vec,channel), total_duration=tTotal)
    return Pattern, state_vec



if __name__ == '__main__':
    
    #==============================================================================
    # testing block
    #==============================================================================

    ### testing programmatic creation    
    # case 1 low low
    sv1 = np.array([0,0,0,0,1,1,0,1,0,0],dtype='bool')
    
    # case 2 high low
    sv2 = np.array([1,1,0,0,1,1,0,1,0,0],dtype='bool')
    
    # case 3 low high
    sv3 = np.array([0,0,1,0,1,1,0,0,1,1],dtype='bool')
    
    # case 4 high high
    sv4 = np.array([1,1,0,0,0,1,0,0,1,1],dtype='bool')
    
    state_vecs = [sv1,sv2,sv3,sv4]
    
    Pulses = [States2RIOpulses(state_vec,0) for state_vec in state_vecs]
    Patterns = [RIOpattern(name='test',Pulses=P,total_duration=10) for P in Pulses]
    S = RIOsequence('test',Patterns=Patterns)
    
    print("testing all 4 cases sequence creation")
    for i in range(4):
        print(np.all(S.Patterns[i].calc_states(0) == state_vecs[i]))
    print("if all True, test passed") 
    print() 
    
    ### test reading writing
    print("reading and writing tests")
    examples_path = os.path.join(os.path.dirname(os.path.abspath('__file__')),'examples')
    S = read_sqc(os.path.join(examples_path,'tmp.sqc'))
    S.write_sqc(os.path.join(examples_path,'tmp_written.sqc'))
    # test on disk
    if os.name == 'posix':
        print("comparing files on disk")
        retcode = os.system('diff -b ./examples/tmp.sqc ./examples/tmp_written.sqc')
        if retcode == 0:
            print("seqs are identical on disk (ignoring whitespaces)")
        else:
            print("files differ!")
            
    # test in mem
    print("comparing seqs in mem")
    Sr = read_sqc(os.path.join(examples_path,'tmp_written.sqc'))
    if S._Header + S._Lines == Sr._Header + Sr._Lines:
        print("seqs are identical in memory")
    
    ### test random
    P1 = calc_random_pattern_exponential(100,1000,5000,7000,0,'test exp')[0]
    P2 = calc_random_pattern_blocks(100,0.5,1000,5000,7000,1,'test tCorr')[0]
    PA = compose_Patterns([P1,P2],'comp')
    S1 = RIOsequence(seq_name='test exp tcorr',Patterns=[P1,P2,PA])
    S2 = RIOsequence(seq_name='test tcorr exp',Patterns=[P2,P1,PA])
    Sp = randomize_Patterns([P1,P2,PA],nReps=3,seq_name='permute testing')[0]
    
    S1.write_sqc('S1.sqc')
    S2.write_sqc('S2.sqc')
    Sp.write_sqc('Sp.sqc')
    
    ### testing vis
    P0,sv0 = calc_random_pattern_exponential(100,1000,5000,7000,0,'stim exp A')
    P1,sv1 = calc_random_pattern_exponential(100,1000,5000,7000,1,'stim exp B')
    P3 = RIOpattern(name='bal B', Pulses=States2RIOpulses(sv1,3), total_duration=7000)
    P4 = RIOpattern(name='bal A', Pulses=States2RIOpulses(sv0,4), total_duration=7000)
    P5 = RIOpattern(name='pre A', Pulses=[RIOpulse(5,1,0,6000)])
    P6 = RIOpattern(name='pre B', Pulses=[RIOpulse(6,1,0,6000)])
    P8 = RIOpattern(name='trigger',Pulses=[RIOpulse(8,1,1000,50)])
    PC = compose_Patterns([P0,P1,P3,P4,P5,P6,P8],'composition test',total_duration=7000)
    PC.preview_plot()

    ### testing vis
    P0,sv0 = calc_random_pattern_exponential(100,1000,5000,7000,0,'stim exp A')
    P1,sv1 = calc_random_pattern_exponential(100,1000,5000,7000,1,'stim exp B')
    PC = compose_Patterns([P0,P1],'composition test',total_duration=7000)
#    PC.preview_plot()