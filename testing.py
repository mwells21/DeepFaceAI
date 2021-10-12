import tensorflow as tf
import cv2  # Open cv
import os
import matplotlib.pyplot as plt
import numpy as np


model = tf.keras.models.load_model("model_15.h5")