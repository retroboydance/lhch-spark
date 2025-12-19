import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

DS_PATH = os.path.dirname(os.path.realpath(__name__))

class Dataset:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_frame = self.open_dataset_csv(dataset_name)
        self.ref = set(self.data_frame['idx'])
        self.indiv_item = dict(zip(self.ref, set(self.data_frame['desc'])))
        self.responses = {}

    def open_dataset_csv(self, dataset_name: str):
        assert dataset_name in ['bcis']
        filename: str = dataset_name + r'-items.csv'
        path: Path = Path(DS_PATH, filename)
        return pd.read_csv(path)
    
    def itemise(self):
        self.data_frame.index = self.data_frame['idx']
        for item in self.ref:
            try:
                self.responses[item] = set(self.data_frame.loc[item]['long'].to_list())
            except:
                print(f'\n{item}: Break')
                break

if __name__ == '__main__':
    ds = Dataset('bcis')