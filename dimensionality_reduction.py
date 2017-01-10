import numpy as np
from sklearn.decomposition import PCA


def run():
    cell_states = np.loadtxt('cell_states.csv', delimiter=',')
    pca = PCA(n_components=50)
    pca.fit(cell_states)
    reduced = pca.transform(cell_states)
    np.savetxt('reduced_states.csv', reduced, delimiter=',', fmt='%.4f')

if __name__ == '__main__':
    run()

