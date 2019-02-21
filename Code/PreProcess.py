import numpy as np


def normalize(signal):
    s = signal.copy()
    s /= np.abs(s).max()
    return s
