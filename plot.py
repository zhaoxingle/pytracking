# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist

trackers = []
# trackers.extend(trackerlist('atom', 'default', range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', range(0,5), 'PrDiMP50'))
trackers.extend(trackerlist('tomp', 'tomp50', range(0,1), 'TOMP50'))

#dataset = get_dataset('otb')T
dataset=get_dataset('satsot')
plot_results(trackers, dataset, 'SATSOT_TOMP50', merge_results=True, plot_types=('success', 'prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)

# filter_criteria = {'mode': 'ao_max', 'threshold': 40.0}
# print_per_sequence_results(trackers, dataset, 'SATSOT_tomp101', merge_results=True, filter_criteria=filter_criteria, force_evaluation=False)