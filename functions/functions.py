alphabet_dictionary = {"A": 11, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18, "I": 19, "J": 20,
                       "K": 21, "L": 22, "M": 23, "N": 24, "O": 25, "P": 26, "Q": 27, "R": 28, "S": 29, "T": 30,
                       "U": 31, "V": 32, "W": 33, "X": 34, "Y": 35, "Z": 36}


def p_spectrum_kernel_function(string1, string2):
    kernel_matrix = []
    for ele1 in string1:
        row = []
        for ele2 in string2:
            kernel = 0
            for i in ele1:
                for j in ele2:
                    if i == j and i != 0:
                        kernel += 1
            row.append(kernel)
        kernel_matrix.append(row)
    return kernel_matrix


def get_substring(string, spectrum=3):
    if spectrum == 0:
        result = ['']
    elif len(string) <= spectrum:
        result = [string]
    else:
        result = [string[i: i + spectrum] for i in range(len(string) - spectrum + 1)]
    return result


def from_string_to_int(string):
    result = ''
    for element in string:
        result += str(alphabet_dictionary[element])
    return int(result)


def pad_data(X_train):

    max_size = 0
    for i in X_train:
        size = len(i)
        if size > max_size:
            max_size = size

    # ...rendo tutti gli altri della stessa taglia

    for i in X_train:
        if len(i) != max_size:
            for j in range(len(i), max_size):
                i.append(0)
    return X_train, max_size

if __name__ == '__main__':
    s = 'jhgfdkhgv'

