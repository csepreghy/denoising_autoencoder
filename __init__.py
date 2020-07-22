from tensorflow.keras.optimizers import SGD
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

import os
import cv2
import matplotlib.pyplot as plt

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

    return images

images = get_images()

print(f'images = {images}')

# response = requests.get(url)
# img = Image.open(BytesIO(response.content))
# img.load()
# img = img.resize((128,128), Image.ANTIALIAS)
# img_array = np.asarray(img)
# img_array = img_array.flatten()
# img_array = np.array([ img_array ])
# img_array = img_array.astype(np.float32)
# print(img_array.shape[1])
# print(img_array)

# model = Sequential()
# model.add(Dense(10, input_dim=img_array.shape[1], activation='relu'))
# model.add(Dense(img_array.shape[1])) # Multiple output neurons
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(img_array,img_array,verbose=0,epochs=20)

# print("Neural network output")
# pred = model.predict(img_array)
# print(pred)
# print(img_array)
# cols,rows = img.size
# img_array2 = pred[0].reshape(rows,cols,3)
# img_array2 = img_array2.astype(np.uint8)
# img2 = Image.fromarray(img_array2, 'RGB')
# img2.show()