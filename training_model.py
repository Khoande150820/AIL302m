from Layer import *
from Accuracy import *
from Loss import *
from Optimizer import *
import numpy as np
from Activation import *
from tqdm import tqdm
import cv2
import os
import pickle, random
from Model import *


def get_data():
    train_size = 19957

    with open("data/X.pickle", "rb") as f:
        X = pickle.load(f)
    with open("data/y.pickle", "rb") as f:
        y = pickle.load(f)
    y = y.reshape(-1,1)
    X = X / 255
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = get_data()

model = Model()
model.add(Layer_Dense(X_train.shape[1], 128))
model.add(Activation_Sigmoid())
model.add(Layer_Dense(128, 128))
model.add(Activation_Sigmoid())
model.add(Layer_Dense(128, 1))
model.add(Activation_Sigmoid())

model.set(loss=Loss_BinaryCrossentropy(),
          optimizer=Optimizer_Adam(decay=1e-3),
          accuracy=Accuracy_Categorical())
model.finalize()





model.train(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

model.save("model.model")