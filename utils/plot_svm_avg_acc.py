import matplotlib.pyplot as plt

if __name__ == '__main__':
    ngram_lengths = [2, 3, 5, 7, 10, 15, 25]
    accuracies = [0.68, 0.695, 0.76, 0.72, 0.72, 0.745, 0.75]

    plt.figure()
    plt.plot(ngram_lengths, accuracies)
    plt.ylim([0, 1])
    plt.xlabel('n-grams length')
    plt.ylabel('accuracy')

    plt.show()
