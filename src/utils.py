import pandas as pd
import numpy as np


def minmax_scaler(x, scale=(0, 1)):
    min, max = scale
    x = np.array(list(x))
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    x = x * (max-min) + min
    return x
