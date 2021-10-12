import tensorflow as tf
import cv2  # Open cv
import os
import matplotlib.pyplot as plt
import numpy as np


model = tf.keras.models.load_model("model_15.h5")

# Testing Dir
Datadirectory = "data/test/"  # Training data
Classes = ["0", "1", "2", "3", "4", "5", "6"]


print("PRE PROCESSING ....... ")
training_Data = []

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_Data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_Data()

print(len(training_Data))

import random
print("---- Shuffling ----")
random.shuffle(training_Data)

X = []
Y = []

print("---- Appending ----")
for features, label in training_Data:
    X.append(features)
    Y.append(label)

X = X[1:200]
Y = Y[1:200]



# 4d
print("---- Reshape ----")
X = np.array(X,dtype = 'uint8').reshape(-1, img_size, img_size, 3)
Y = np.array(Y,dtype= 'uint8')

# Normalize
print("---- Normalize ----")
X = X/255.0