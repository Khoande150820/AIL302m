from Layer import *
from Accuracy import *
from Loss import *
from Optimizer import *
import numpy as np
from tqdm import tqdm
import cv2
import os
import pickle, random
from Model import Model
import matplotlib.pyplot as plt
DATA_DIR = "D:\\FPT_project\\AILm\\Project\\Data\\PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50


def create_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)  # Create path to file
        class_num = CATEGORIES.index(category)  # get the classification 0 is dog, 1 is cat

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert image to gray scale
                img_array_resize = cv2.resize(img_array, (50, 50))
                training_data.append([img_array_resize, class_num])
            except Exception as e:
                print(f"Error at {os.path.join(path, img)}"
                      f"\n"

                      f"Detail: {e}")
    random.shuffle(training_data)
    return training_data


def save_data(training_data):
    X = []
    y = []
    for feature, label in training_data:
        X.append(feature)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    pickle_out = open("data/X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open("data/y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


TEST_DIR = "TestImg"
PREDICT_LABEL = {0: "dog", 1: "cat"}


def get_prediction(model):

    for img in os.listdir(TEST_DIR):
        plt.figure()
        print(img)
        raw_img = plt.imread(os.path.join(TEST_DIR,img))
        X = cv2.resize(cv2.imread(os.path.join(TEST_DIR, img), cv2.IMREAD_GRAYSCALE), (50, 50))
        X = np.array(X)
        X = X / 255
        X = X.reshape(1, -1)
        confidences = model.predict(X)
        predictions = model.output_layer_activation.predictions(confidences)
        predictions = predictions.reshape(-1)
        predictions = int(predictions)
        label = PREDICT_LABEL[predictions]
        plt.imshow(raw_img)
        plt.legend(label)



if __name__ == '__main__':
    model = Model.load("model.model")
    get_prediction(model)