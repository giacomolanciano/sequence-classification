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
SPECTRUM_KEY = 'spectrum'
LABELS_KEY = 'labels'
INPUTS_PER_LABEL_KEY = 'ipl'
TIME_KEY = 'time'
TRAIN_DATA_KEY = 'train_data'
TEST_DATA_KEY = 'test_data'
TRAIN_LABELS_KEY = 'train_labels'
TEST_LABELS_KEY = 'test_labels'
TRAIN_DATA_POS = 0
TEST_DATA_POS = 1
TRAIN_LABELS_POS = 2
TEST_LABELS_POS = 3


if __name__ == '__main__':
    print(DATA_FOLDER)
    print(DATABASE)
