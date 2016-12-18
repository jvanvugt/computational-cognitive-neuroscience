"""
Author: Joris van Vugt

A numpy implementation of an LSTM followed by
a dense classification layer. This allows for
easier modifications and monitoring of the activations
and network state
"""
import numpy as np
import h5py


def softmax(x):
    """
    The softmax activation function
    Normalize log probabilities
    """
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1)

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """

    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    a_i, a_f, a_o, a_g = np.split(a, 4, axis=1)
    i = sigmoid(a_i)
    f = sigmoid(a_f)
    o = sigmoid(a_o)
    g = np.tanh(a_g)
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    return next_h, next_c

def create_lstm_matrices(filepath, layer):
    """
    Load the weight matrices from an LSTM in
    hdf5 format. The weight matrices are concatenated
    because keras splits them up.
    """
    prefix = '{0}/{0}_'.format(layer)
    with h5py.File(filepath) as f:
        Wx = np.hstack((f[prefix + 'W_i:0'][:],
                        f[prefix + 'W_f:0'][:],
                        f[prefix + 'W_o:0'][:],
                        f[prefix + 'W_c:0'][:]))
        Wh = np.hstack((f[prefix + 'U_i:0'][:],
                        f[prefix + 'U_f:0'][:],
                        f[prefix + 'U_o:0'][:],
                        f[prefix + 'U_c:0'][:]))
        b =  np.hstack((f[prefix + 'b_i:0'][:],
                        f[prefix + 'b_f:0'][:],
                        f[prefix + 'b_o:0'][:],
                        f[prefix + 'b_c:0'][:]))
    return Wx, Wh, b

def create_dense_matrices(filepath, layer):
    """
    Load the weight matrices from a
    time distributed dense layer in hdf5 format
    """
    prefix = layer + '/dense_1_'
    with h5py.File(filepath) as f:
        W = f[prefix + 'W:0'][:]
        b = f[prefix + 'b:0'][:]
        return W, b


def load_data(filepath):
    """
    Load text from a text file and create
    look-up tables
    """
    with open(filepath) as file:
        text = file.read()[:2000000]
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return text, char_indices, indices_char

def sample(preds, temperature=1.0):
    """
    helper function to sample an index from a probability array
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(initial_char='d', timesteps=400, temperature=1., save_cells=True):
    """
    Generate text by doing *timesteps* forward passes through
    the network. At each timestep, a character is sampled.
    initial_char is used as a seed. temperature controls
    the variability in the output. Low temperature will
    lead to repetetive results(<0.5), high temperature (>1.2) might
    cause random garbage as output.
    The result is printed.
    """
    # Load the weights
    Wx, Wh, b = create_lstm_matrices('weights.hdf5', 'lstm_1')
    Wd, bd = create_dense_matrices('weights.hdf5', 'timedistributed_1')
    _, char_indices, indices_char = load_data('D:/Data/python.txt')

    # Set up the initial state
    x = np.zeros((timesteps+1, len(char_indices)))
    x[0, char_indices[initial_char]] = 1
    H = Wh.shape[0]
    h = np.zeros((timesteps+1, H))
    c = np.zeros_like(h)
    buffer = ''

    # Forward passes through the model
    for t in range(timesteps):
        h[t+1, :], c[t+1, :] = lstm_step_forward(x[t, :].reshape(1, -1), h[t, :], c[t, :], Wx, Wh, b)
        logprobs = np.dot(h[t+1], Wd) + bd
        probs = softmax(logprobs.reshape(1, -1))[0]
        next_index = sample(probs, temperature)
        buffer += indices_char[next_index]
        x[t+1, next_index] = 1
    
    if save_cells:
        np.savetxt('cell_states.csv', c, delimiter=',', fmt='%.4f')
    
    with open('generated.txt', 'w') as file:
        file.write(buffer)
    print(buffer)





if __name__ == '__main__':
    generate_text()