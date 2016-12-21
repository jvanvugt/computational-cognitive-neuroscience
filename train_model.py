'''
Adapted version of the Keras LSTM text generation example
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback, TensorBoard
from keras.optimizers import RMSprop
import numpy as np
import sys
import random
import pickle

# Resume from checkpoint?
resume = False

path = 'D:/Data/python.txt'
text = open(path).read()
# Unix includes '\r' when reading a file while Windows doesn't. 
# We remove it to avoid inconsistencies.
text = text.replace('\r', '')
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 100
step = 31 # Prime number to avoid patterns
sentences = []
offset_sentences = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    offset_sentences.append(text[i+1: i+maxlen+1])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
for i, (sentence, off_sentence) in enumerate(zip(sentences, offset_sentences)):
    for t, (char, next_char) in enumerate(zip(sentence, off_sentence)):
        X[i, t, char_indices[char]] = 1
        y[i, t, char_indices[next_char]] = 1

print('Build model...')
model = Sequential()
model.add(LSTM(1024, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=2e-3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()


def schedule(epoch):
    lr = 2e-3 if epoch <= 9 else 2e-3 * 0.95**(epoch-8)
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

        sentence = text[start_index: start_index + maxlen]
        generated = sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0, -1, :]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

checkpointer = ModelCheckpoint('weights.hdf5', save_best_only=True, save_weights_only=True)
lr_schedule = LearningRateScheduler(schedule)
sampler = LambdaCallback(on_epoch_end=lambda _, __: generate_text())
tensorboard = TensorBoard()
if resume:
    model.load_weights('weights.hdf5')


# train the model, output generated text after each iteration
history = model.fit(X, y, batch_size=128, nb_epoch=50, validation_split=0.1,
                    callbacks=[checkpointer, lr_schedule, sampler, tensorboard])
with open('history.pkl', 'w') as file:
    pickle.dump(history, file)
