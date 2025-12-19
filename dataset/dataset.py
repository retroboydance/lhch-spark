import os
import pandas as pd
import numpy as np
import pickle

DS_PATH = os.path.dirname(os.path.realpath(__name__))

if __name__ == '__main__':
    ds = pd.read_csv(DS_PATH)