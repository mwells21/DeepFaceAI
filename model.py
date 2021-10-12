import tensorflow as tf
import cv2  # Open cv
import os
import matplotlib.pyplot as plt
import numpy as np

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

# new_model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam", metrics = ["accuracy"])