import numpy as np

def log_decay(start, end, length):
    return np.logspace(np.log10(start), np.log10(end), length)