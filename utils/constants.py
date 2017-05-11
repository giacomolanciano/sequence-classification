""" Module containing all constants. """
import os

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
TRAINED_MODELS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models')
TF_MODEL_EXT = '.ckpt'
DATABASE = os.path.join(DATA_FOLDER, 'proteins.db')
PADDING_VALUE = 0
FILENAME_SEPARATOR = '_'
RNN_SUFFIX = 'rnn'
SPECTRUM_SUFFIX = 'spectrum'
DUMP_EXT = '.pickle'
SPECTRUM_KEY = 'spectrum'
LABELS_KEY = 'labels'
INPUTS_PER_LABEL_KEY = 'ipl'
TIME_KEY = 'time'
DATASET_KEY = 'dataset'


if __name__ == '__main__':
    print(DATA_FOLDER)
    print(DATABASE)
