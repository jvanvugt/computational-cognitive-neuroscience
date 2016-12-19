import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.5

def plot_saturation():
    data = np.loadtxt('cell_states.csv', delimiter=',')
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
    plot_saturation()