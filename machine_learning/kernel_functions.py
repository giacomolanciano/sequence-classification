from collections import Counter

from utils.constants import PADDING_VALUE


def naive_spectrum_kernel(X, Y):
    kernel_matrix = []
    for shingles_list_1 in X:
        row = []
        for shingles_list_2 in Y:
            kernel = 0
            for shingle in shingles_list_1:
                for j in shingles_list_2:
                    if shingle != PADDING_VALUE and shingle == j:
                        kernel += 1
            row.append(kernel)
        kernel_matrix.append(row)
    return kernel_matrix


def occurrence_dict_spectrum_kernel(X, Y):
    kernel_matrix = []
    for shingles_list_1 in X:
        row = []
        shingles_list_1_dict = Counter(shingles_list_1)
        for shingles_list_2 in Y:
            kernel = 0
            shingles_list_2_dict = Counter(shingles_list_2)
            for shingle, occurrences in shingles_list_1_dict.items():
                if shingle != PADDING_VALUE:
                    try:
                        kernel += shingles_list_2_dict[shingle] * occurrences
                    except KeyError:
                        continue
            row.append(kernel)
        kernel_matrix.append(row)
    return kernel_matrix
