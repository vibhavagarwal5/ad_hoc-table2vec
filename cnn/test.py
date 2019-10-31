import pickle
import numpy as np

with open('X_2.pkl', 'rb') as f:
    unpickler = pickle.Unpickler(f)
    X = unpickler.load()
    # X = pickle.load(f)
with open('y_2.pkl', 'rb') as f:
    y = pickle.load(f)
print(len(list(sum(X,[]))))
print(len(list(sum(y,[]))))
