from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# setting the path of directories
train_path = "covid_data/train"
categories = ["covid", "normal"]

training_data = []
for category in categories:
    path = os.path.join(train_path, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (256, 256))
        training_data.append([new_array, class_num])

print("Training data length", len(training_data))
x = []
y = []
for features, label in training_data:
    x.append(features)
    y.append(label)
x_train = np.array(x).reshape(2883, 256, 256, 1)
x_train = x_train/255.0
y_train = np.array(y)


(trainX, textX, trainY, testY) = train_test_split(x_train, y_train, test_size=0.20, stratify=y_train, random_state=42)
trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
baseModel = VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(224, 224, 3)))
