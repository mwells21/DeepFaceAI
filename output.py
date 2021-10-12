import tensorflow as tf
import cv2  # Open cv
import os
import matplotlib.pyplot as plt
import numpy as np


model = tf.keras.models.load_model("model_15.h5")

# Test with image

frame = cv2.imread("happy.jpg")

plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2RGB))