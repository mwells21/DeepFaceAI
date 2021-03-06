import tensorflow as tf
import cv2  # Open cv
import os
import matplotlib.pyplot as plt
import numpy as np


model = tf.keras.models.load_model("model_20_test.h5")

# Testing Dir
Datadirectory = "data/train/"  # Training data
Classes = ["0", "1", "2", "3", "4", "5", "6"]

img_size = 224


print("PRE PROCESSING ....... ")
testing_Data = []

def create_testing_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                testing_Data.append([new_array, class_num])
            except Exception as e:
                pass


create_testing_Data()

print(len(testing_Data))

import random
print("---- Shuffling ----")
random.shuffle(testing_Data)

X = []
Y = []

print("---- Appending ----")
for features, label in testing_Data:
    X.append(features)
    Y.append(label)

X = X[1:1000]
Y = Y[1:1000]



# 4d
print("---- Reshape ----")
X = np.array(X,dtype = 'uint8').reshape(-1, img_size, img_size, 3)
Y = np.array(Y,dtype= 'uint8')

# Normalize
print("---- Normalize ----")
X = X/255.0

# Test Set
print("Testing.....")
results = model.evaluate(X, Y)
print("test loss, test acc:", results)