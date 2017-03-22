""" Module containing all constants. """
import os

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DATABASE = os.path.join(DATA_FOLDER, 'proteinDB.db')
PADDING_VALUE = 0


if __name__ == '__main__':
    print(DATA_FOLDER)
    print(DATABASE)
