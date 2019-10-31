import pandas as pd
import numpy as np
import pickle
import random
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D, Embedding
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 20
batch_size = 32
embedding_dims = 100
hidden_dims = 250
epochs = 2

def get_data(X_f, y_f):
    with open(X_f, 'rb') as f:
#         unpickler = pickle.Unpickler(f)
#         X = unpickler.load()
        X = pickle.load(f)
    with open(y_f, 'rb') as f:
        y = pickle.load(f)
    X = list(sum(X, []))
    y = list(sum(y, []))
    return X,y

def save_np(f, inp):
    np.save(f, inp)

def load_np(f):
    return np.load(f)

def gen_vocab(X):
    vocab = []
    for i in X:
        for j in i:
            vocab.append(j)
    return np.array(list(set(vocab)))

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

if __name__ == "__main__":
    X, y = get_data('./X_l20.pkl', './y_l20.pkl')
    # X = pad_seq(X)
    X = np.array(X)
    X = encode_text(create_tokenizer(X), X, maxlen)
    y = np.array(y).reshape((-1,1))
    # save_np('X_2_emb.npy', X)
    # save_np('y_2_emb.npy', y)

    # X = load_np('X_2_emb.npy')
    # y = load_np('y_2_emb.npy')

    print(X.shape,y.shape)

    vocab = gen_vocab(X)
    print(vocab.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = Sequential()
    model.add(Embedding(vocab.shape[0], embedding_dims,input_length=maxlen))
    model.add(Conv1D(64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])