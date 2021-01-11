import os
import numpy as np
import cv2
import pickle
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, LeakyReLU
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# path setting

train_path = "dataset/train"
valid_path = "dataset/valid"
categories = ["Covid19", "No_findings", "Pneumonia"]
x_train1 = []
y_train1 = []
x_valid1 = []
y_valid1 = []


def train_data_collection():
    for category in categories:
        path = os.path.join(train_path, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img1 = load_img(os.path.join(train_path, img), color_mode="grayscale", target_size=(256, 256))
            img = img_to_array(img1)
            x_train1.append(img)
            y_train1.append(class_num)


train_data_collection()


def valid_data_collection():
    for category in categories:
        path = os.path.join(valid_path, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            img1 = load_img(os.path.join(valid_path, img), color_mode="grayscale", target_size=(256, 256))
            img = img_to_array(img1)
            x_valid1.append(img)
            y_valid1.append(class_num)


valid_data_collection()
x_train = np.array([x_train1]).reshape(256, 1)
print(x_train)
model = Sequential()


def conv_block(ni, size=3):
    model.add(Conv2D(ni, (size, size), strides=(1, 1), use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))


def triple_conv(ni, nf):
    conv_block(ni)
    conv_block(nf, size=1)
    conv_block(ni)


model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(2, 2))
conv_block(8, 16)
model.add(MaxPool2D(2, 2))
triple_conv(16, 32)
model.add(MaxPool2D(2, 2))
triple_conv(32, 64)
model.add(MaxPool2D(2, 2))
triple_conv(64, 128)
model.add(MaxPool2D(2, 2))
triple_conv(128, 256)
conv_block(256, 128)
conv_block(128, 256)
model.add(Conv2D(256, (3, 3)))
model.add(Flatten())
model.add(Dense(3, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

model.fit(x=x_train, y=y_train1, epochs=3)
