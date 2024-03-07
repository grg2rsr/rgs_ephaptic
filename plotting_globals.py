
import matplotlib as mpl

page_width = 180 # in mm

SMALLEST_SIZE = 4
SMALL_SIZE = 5
MEDIUM_SIZE = 6

mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
mpl.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

mpl.rc('figure', dpi=300)