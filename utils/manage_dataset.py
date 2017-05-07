import os

import pickle
from pprint import pprint

from utils.constants import DATA_FOLDER


def dump_dataset(dataset_dict, filename):
    """
    Create a dump (named as 'filename') of the given dataset in secondary storage.
    
    :type dataset_dict: dict
    :type filename: str
    :param dataset_dict: a dict representing the feature of the dataset.
    :param filename: the filename of the dataset dump.
    """
    filename = os.path.join(DATA_FOLDER, filename)
    with open(filename, 'wb') as data_dump:
        pickle.dump(dataset_dict, data_dump)


def load_dataset(cached_dataset):
    """
    Load a dataset in memory from a dump in secondary storage identified by the given filename.
    
    :type cached_dataset: str
    :param cached_dataset: the filename of the dataset.
    :return: the object that represents the dataset.
    """
    filename = os.path.join(DATA_FOLDER, cached_dataset)
    with open(filename, 'rb') as spilt_dataset:
        return pickle.load(spilt_dataset)

if __name__ == '__main__':
    # insert the code for the desired manipulation of a cached dataset.
    d = load_dataset(cached_dataset='1494155537.5589833_3_OXIDOREDUCTASE_PROTEIN TRANSPORT_rnn.pickle')
    td = d['dataset'][0]
    pprint(td[0])
