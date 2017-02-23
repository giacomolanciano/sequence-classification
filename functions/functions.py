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
                    if (i == j) and (i != 0):
                        kernel += 1
            row.append(kernel)
        kernel_matrix.append(row)

    print(kernel_matrix)
    return kernel_matrix

def get_substring(m_string, spectrum):
    tmp_list = []
    if spectrum == 0:
        tmp_list = ['']
    else:
        for i in range(len(m_string) - spectrum + 1):
            m_string_res = ''
            for j in range(spectrum):
                m_string_res += m_string[i + j]
            tmp_list.append(m_string_res)
    return tmp_list


def f_s_t_f(m_string):
    result = ""
    for element in m_string:
        result += str(alphabet_dictionary[element])
    return int(result)
