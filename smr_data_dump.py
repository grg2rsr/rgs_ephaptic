# os 
import os
import sys
from pathlib import Path

# sci
import numpy as np

# ephys
import neo
sys.path.append('/home/georg/code/SSSort')
import sssio

# plotting
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = Path(sys.argv[1])
    print("dumping %s" % path.stem)

    seg = sssio.smr2seg(path, channel_index=1)
    asig = seg.analogsignals[0]
    data = np.stack([asig.times.magnitude, asig.flatten().magnitude]).T

    np.save(path.with_suffix('.npy'), data.astype(asig.dtype))
    np.save(path.parent / (path.stem + '_trial_times.npy'), seg.events[0].times.magnitude)