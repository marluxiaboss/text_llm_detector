import logging
from time import strftime, gmtime
import sys

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


def compute_bootstrap_metrics(data, labels, n_bootstrap=1000):

    # compute false postives, false negatives, true positives, true negatives using bootstrap
    nb_false_positives = np.zeros(n_bootstrap)
    nb_false_negatives = np.zeros(n_bootstrap)
    nb_true_positives = np.zeros(n_bootstrap)
    nb_true_negatives = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(range(len(data)), len(data), replace=True)
        nb_false_positives[i] = np.sum((data[bootstrap_sample] == 1) & (labels[bootstrap_sample] == 0))
        nb_false_negatives[i] = np.sum((data[bootstrap_sample] == 0) & (labels[bootstrap_sample] == 1))
        nb_true_positives[i] = np.sum((data[bootstrap_sample] == 1) & (labels[bootstrap_sample] == 1))
        nb_true_negatives[i] = np.sum((data[bootstrap_sample] == 0) & (labels[bootstrap_sample] == 0))
    
    metrics = ["accuracy", "precision", "recall", "f1_score", "fp_rate"]
    avg_metrics = {}
    std_metrics = {}
    for metric in metrics:
        metric_results = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            nb_false_positives_i = nb_false_positives[i]
            nb_false_negatives_i = nb_false_negatives[i]
            nb_true_positives_i = nb_true_positives[i]
            nb_true_negatives_i = nb_true_negatives[i]

            if metric == "accuracy":
                metric_results[i] = (nb_true_positives_i + nb_true_negatives_i) / len(data)
            elif metric == "precision":
                metric_results[i] = nb_true_positives_i / (nb_true_positives_i + nb_false_positives_i)
            elif metric == "recall":
                metric_results[i] = nb_true_positives_i / (nb_true_positives_i + nb_false_negatives_i)
            elif metric == "f1_score":
                metric_results[i] = 2 * nb_true_positives_i / (2 * nb_true_positives_i + nb_false_positives_i + nb_false_negatives_i)
            elif metric == "fp_rate":
                metric_results[i] = nb_false_positives_i / (nb_false_positives_i + nb_true_negatives_i)

        avg_metrics[metric] = np.mean(metric_results)
        std_metrics[metric] = np.std(metric_results)

    print("Average metrics: ", avg_metrics)
    print("Standard deviation of metrics: ", std_metrics)

    # change name of std_metrics as std_{metric_name}
    for metric in metrics:
        std_metrics["std_" + metric] = std_metrics[metric]
        del std_metrics[metric]
    
    avg_metrics.update(std_metrics)
    metrics_dict = avg_metrics
    return metrics_dict

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
        
    