"""
Author: Joris van Vugt

Plot the activations and cell states of neurons in different ways
"""
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.5

def load_data(filename):
    """
    Loads the cell states from the specified file
    """
    return np.loadtxt(filename, delimiter=',')

def plot_correlated(threshold=0.8):
    """
    Computes the correlation between each pair of neuros

    Neurons which have an absolute correlation higher
    than the specified threshold are plotted.
    """
    data = load_data('cell_states.csv')
    corrs = np.corrcoef(data, rowvar=0)
    n_neurons = corrs.shape[0]
    t = data.shape[0]

    large_corrs = []
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if np.abs(corrs[i, j]) > threshold:
                large_corrs.append((i, j))
    print('Found %d neurons with abs(correlation) > %.2f' % (len(large_corrs), threshold))
    
    for i, j in large_corrs:
        plt.plot(data[:, i], label='Neuron %d' % i, c='r')
        plt.plot(data[:, j], label='Neuron %d' % j, c='b')
        plt.plot([0, t], [0, 0], '--', c='k')
        y_min = min(data[:, i].min(), data[:, j].min())
        plt.text(10, y_min, '$r=%.2f$' % corrs[i, j], {'size': 22})
        plt.xlim([0, t])
        plt.legend()
        plt.show()


def plot_saturation():
    data = load_data('cell_states.csv')
    t = data.shape[0]
    layer1_pos = (np.tanh(data[:, :1024]) >= 1 - EPSILON).sum(axis=0) / t
    layer1_neg = (np.tanh(data[:, :1024]) < -1 + EPSILON).sum(axis=0) / t
    
    layer2_pos = (np.tanh(data[:, 1024:]) >= 1 - EPSILON).sum(axis=0) / t
    layer2_neg = (np.tanh(data[:, 1024:]) < -1 + EPSILON).sum(axis=0) / t

    plt.scatter(layer1_pos, layer1_neg, s=60, alpha=0.3, c='r', label='Layer 1')
    plt.scatter(layer2_pos, layer2_neg, s=60, alpha=0.3, c='b', label='Layer 2')
    plt.plot(np.linspace(1, 0, 2), '--', c='k')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Fraction right saturated')
    plt.ylabel('Fraction left saturated')
    plt.legend()
    plt.savefig('../figures/saturation.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_correlated()