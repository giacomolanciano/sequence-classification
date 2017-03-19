from collections import Counter

from machine_learning.sequence_classifier import SequenceClassifierInput
from utils.constants import PADDING_VALUE


def naive_spectrum_kernel(rows_data, cols_data):
    kernel_matrix = []
    for rows_shingles_list in rows_data:
        row = []
        for cols_shingles_list in cols_data:
            kernel = 0
            for row_shingle in rows_shingles_list:
                for col_shingle in cols_shingles_list:
                    if row_shingle != PADDING_VALUE and row_shingle == col_shingle:
                        kernel += 1
            row.append(kernel)
        kernel_matrix.append(row)
    return kernel_matrix


def occurrence_dict_spectrum_kernel(rows_data, cols_data):
    kernel_matrix = []
    for rows_shingles_list in rows_data:
        row = []
        rows_shingles_occ_dict = Counter(rows_shingles_list)
        for cols_shingles_list in cols_data:
            kernel = 0
            cols_shingles_occ_dict = Counter(cols_shingles_list)
            for shingle, occurrences in rows_shingles_occ_dict.items():
                if shingle != PADDING_VALUE:
                    try:
                        kernel += cols_shingles_occ_dict[shingle] * occurrences
                    except KeyError:
                        continue
            row.append(kernel)
        kernel_matrix.append(row)
    return kernel_matrix


def precomputed_occurrence_dict_spectrum_kernel(data):
    data_size = len(data)
    data_size_range = range(data_size)
    kernel_matrix = [[0] * data_size for _ in data_size_range]

    for row_num in data_size_range:
        rows_shingles_occ_dict = Counter(data[row_num])
        for col_num in range(row_num, data_size):
            kernel = 0
            cols_shingles_occ_dict = Counter(data[col_num])
            for shingle, occurrences in rows_shingles_occ_dict.items():
                if shingle != PADDING_VALUE:
                    try:
                        kernel += cols_shingles_occ_dict[shingle] * occurrences
                    except KeyError:
                        continue
            kernel_matrix[row_num][col_num] = kernel
            kernel_matrix[col_num][row_num] = kernel
    return kernel_matrix


if __name__ == '__main__':
    import time
    from datetime import timedelta

    clf_input = SequenceClassifierInput(inputs_per_label=100)
    clf_input.set_train_test_data(['TRANSCRIPTION', 'LYASE'])
    inputs_data = clf_input.train_data + clf_input.test_data

    # test kernels performances on same data
    start_time = time.time()
    kernel_matrix_train = occurrence_dict_spectrum_kernel(inputs_data, inputs_data)
    elapsed_time = (time.time() - start_time)
    print('\nTime: ', timedelta(seconds=elapsed_time))

    start_time = time.time()
    kernel_matrix_train = precomputed_occurrence_dict_spectrum_kernel(inputs_data)
    elapsed_time = (time.time() - start_time)
    print('\nTime: ', timedelta(seconds=elapsed_time))
