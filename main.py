import tensorflow as tf
import cv2  # Open cv
import os
import matplotlib.pyplot as plt
import numpy as np
# ---- Testing ------
img_array = cv2.imread("data/train/0/Training_964885.jpg")
print(img_array.shape)
plt.imshow(img_array)
plt.show()

Datadirectory = "data/train/"  # Training data
Classes = ["0", "1", "2", "3", "4", "5", "6"]

img_size = 224
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array)
plt.show()


# ---- Pre Processing -----
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

X = X[1:1000]
Y = Y[1:1000]



# 4d
print("---- Reshape ----")
X = np.array(X,dtype = 'uint8').reshape(-1, img_size, img_size, 3)
Y = np.array(Y,dtype= 'uint8')

# Normalize
print("---- Normalize ----")
X = X/255.0

# ---- Model ---- 
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.applications.MobileNetV2() # This is the pre-trained model
# model.summary()


base_input = model.layers[0].input

base_output = model.layers[-2].output

final_out = layers.Dense(128)(base_output)
final_out = layers.Activation('relu')(final_out)
final_out = layers.Dense(64)(final_out)
final_out = layers.Activation('relu')(final_out)
final_out = layers.Dense(7,activation='softmax')(final_out) # adjusted for 7 Classes


new_model = keras.Model(inputs = base_input,outputs = final_out)

new_model.summary()

new_model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])

new_model.fit(X,Y,epochs = 15)

new_model.save('model_15.h5')