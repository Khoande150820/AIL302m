
import numpy as np
import cv2
import os

from Model import Model
import matplotlib.pyplot as plt


TEST_DIR = "TestImg"
PREDICT_LABEL = {0: "dog", 1: "cat"}


def get_prediction(model):

    for img in os.listdir(TEST_DIR):
        plt.figure()
        print(img)
        X = cv2.resize(cv2.imread(os.path.join(TEST_DIR, img), cv2.IMREAD_GRAYSCALE), (50, 50))
        X = np.array(X)
        X = X / 255
        X = X.reshape(1, -1)
        confidences = model.predict(X)
        predictions = model.output_layer_activation.predictions(confidences)
        predictions = predictions.reshape(-1)
        predictions = int(predictions)
        label = PREDICT_LABEL[predictions]
        print(label)



if __name__ == '__main__':
    model = Model.load("model.model")
    get_prediction(model)