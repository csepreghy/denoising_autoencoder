from tensorflow.keras.optimizers import SGD
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist

import os
import cv2
import matplotlib.pyplot as plt

(trainX, trainy), (testX, testy) = mnist.load_data()

def save_resized_images():
    datapath = os.path.join(os.getcwd(), 'cats')
    save_path = os.path.join(os.getcwd(), 'cats_resized')
    img_size = 150

    for img_path in os.listdir(datapath):
        img = cv2.imread(os.path.join(datapath, img_path))
        
        if img.shape[0] > img.shape[1]: # if vertical, cut from top and bottom
            pixels_to_cut = int(img.shape[0] * 0.05)
            cropped = img[pixels_to_cut: -pixels_to_cut, 0:-1]
        
        elif img.shape[0] < img.shape[1]: # if horizontal, cut from left and right
            pixels_to_cut = int(img.shape[1] * 0.05)
            cropped = img[0:-1, pixels_to_cut: -pixels_to_cut]
        
        img_resized = cv2.resize(cropped, (img_size, img_size))
        cv2.imwrite(os.path.join(save_path, img_path), img_resized)

def get_images():
    datapath = os.path.join(os.getcwd(), 'cats_resized')
    images = []
    for img_path in os.listdir(datapath):
        img = cv2.imread(os.path.join(datapath, img_path))
        images.append(img)

    return np.array(images)

X = get_images()

# ======================================================================== #
# ================================ DECODER =============================== #
# ======================================================================== #

input_layer = Input(shape=(X.shape[1:]))

# encoder
x = Conv2D(filters=32,
           kernel_size=3,
           activation='relu', 
           padding='same')(input_layer)

x = MaxPooling2D(3)(x)

x = Conv2D(filters=16,
            kernel_size=3,
            activation='relu',
            padding='same')(x)

encoded = MaxPooling2D(2, padding='same')(x)

# ======================================================================== #
# ================================ DECODER =============================== #
# ======================================================================== #

x = Conv2D(filters=16,
           kernel_size=3,
           activation='relu',
           padding='same')(encoded)
        
x = UpSampling2D(2)(x)

x = Conv2D(filters=32,
            kernel_size=3,
            activation='relu',
            padding='same')(x)

x = UpSampling2D(2)(x)

decoded = Conv2D(1, 1, activation='tanh', padding='same')(x)

autoencoder = Model(input_layer, decoded)
