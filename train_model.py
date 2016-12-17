'''
Adapted version of the Keras LSTM text generation example
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback
from keras.layers import LSTM, Dropout
from keras.optimizers import RMSprop
import numpy as np
import sys
import random
import pickle

# Resume from checkpoint?
resume = False

path = 'D:/Data/python.txt'
text = open(path).read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 100
step = 11
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('Build model...')
model = Sequential()
model.add(LSTM(512, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=2e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def schedule(epoch):
    lr = 2e-3 if epoch <= 10 else 2e-3 * 0.95**(epoch-10)
    print('Setting learning rate to', lr)
    return lr


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text():
    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

checkpointer = ModelCheckpoint('weights.hdf5', save_best_only=True, save_weights_only=True)
lr_schedule = LearningRateScheduler(schedule)
sampler = LambdaCallback(on_epoch_end=lambda _, __: generate_text())
if resume:
    model.load_weights('weights.hdf5')


# train the model, output generated text after each iteration
history = model.fit(X, y, batch_size=128, nb_epoch=50, validation_split=0.1,
                    callbacks=[checkpointer, lr_schedule, sampler])
with open('history.pkl') as file:
    pickle.dump(history, file)
