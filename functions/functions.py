from wheel.signatures.djbec import bit

alphabet_dictionary = {"A": 0b10000000000000000000000000,
                       "B": 0b01000000000000000000000000,
                       "C": 0b00100000000000000000000000,
                       "D": 0b00010000000000000000000000,
                       "E": 0b00001000000000000000000000,
                       "F": 0b00000100000000000000000000,
                       "G": 0b00000010000000000000000000,
                       "H": 0b00000001000000000000000000,
                       "I": 0b00000000100000000000000000,
                       "J": 0b00000000010000000000000000,
                       "K": 0b00000000001000000000000000,
                       "L": 0b00000000000100000000000000,
                       "M": 0b00000000000010000000000000,
                       "N": 0b00000000000001000000000000,
                       "O": 0b00000000000000100000000000,
                       "P": 0b00000000000000010000000000,
                       "Q": 0b00000000000000001000000000,
                       "R": 0b00000000000000000100000000,
                       "S": 0b00000000000000000010000000,
                       "T": 0b00000000000000000001000000,
                       "U": 0b00000000000000000000100000,
                       "V": 0b00000000000000000000010000,
                       "W": 0b00000000000000000000001000,
                       "X": 0b00000000000000000000000100,
                       "Y": 0b00000000000000000000000010,
                       "Z": 0b00000000000000000000000001}
#'{0:036b}'.format(1)


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


def encode_sequence(string):
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
