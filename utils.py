import logging
from time import strftime, gmtime
import sys

import numpy as np

def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Create a new logger"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("log/log_%m%d_%H%M.txt", gmtime())
        if type(log_file) == list:
            for filename in log_file:
                fh = logging.FileHandler(filename, mode='w')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                log.addHandler(fh)
        if type(log_file) == str:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)
    return log


def compute_bootstrap_acc(data, n_bootstrap=1000):

    accs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, len(data), replace=True)
        accs[i] = np.mean(bootstrap_sample)

    lower_bound = np.percentile(accs, 2.5)
    upper_bound = np.percentile(accs, 97.5)

    return (np.mean(accs), np.std(accs), lower_bound, upper_bound)



class Signal:
    """Running signal to control training process"""

    def __init__(self, signal_file):
        self.signal_file = signal_file
        self.training_sig = True

        self.update()

    def update(self):
        signal_dict = self.read_signal()
        self.training_sig = signal_dict['training']

    def read_signal(self):
        with open(self.signal_file, 'r') as fin:
            return eval(fin.read())
        
    