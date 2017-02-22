alphabet_dictionary = {"A": 11, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18, "I": 19 ,"J":20 ,"K":21 ,"L": 22,
                       "M": 23, "N": 24, "O": 25, "P": 26, "Q": 27, "R": 28, "S": 29, "T": 30, "U": 31, "V": 32
                       ,"W":33 ,"X":34 ,"Y":35,"Z": 36 }

def pSpectrumKernelFunction(mString1, mString2):
    kernel_matrix = []
    for ele1 in mString1:
        row = []
        for ele2 in mString2:
            kernel = 0
            for i in ele1:
                for j in ele2:
                    if (i == j) and (i != 0):
                        kernel += 1
            row.append(kernel)
        kernel_matrix.append(row)

    print(kernel_matrix)
    return kernel_matrix

def getSubString(mString, spectrum):
    tmpList = []
    if (spectrum == 0):
        tmpList = ['']
    else:
        for i in range(len(mString)-spectrum+1):
            mStringRes = ''
            for j in range(spectrum):
                mStringRes += mString[i+j]
            tmpList.append(mStringRes)
    return tmpList

def f_s_t_f(mystring):
    result = ""
    for element in mystring:
        result = result+str(alphabet_dictionary[element])
    return int(result)