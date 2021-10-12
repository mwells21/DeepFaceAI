import tensorflow as tf
import cv2  # Open cv
import os
import matplotlib.pyplot as plt
import numpy as np


model = tf.keras.models.load_model("model_15.h5")

# Test with image

frame = cv2.imread("happy.jpg")

plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


# import face classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray,1.1,4)

for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame,(x,y),(x+w, y+h), (255, 0, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if(len(facess)) == 0:
        print("Face not detected")
    else:
        for (ex,ey,ew,eh) in facess:
            face_roi = roi_color[ey: ey+eh , ex:ex + ew]


plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(cv2.cvtColor(face_roi,cv2.COLOR_BGR2RGB))
plt.show()

test_image = cv2.resize(face_roi, (224, 224))
test_image = np.expand_dims(test_image,axis = 0)
test_image = test_image/255.0

Predictions = model.predict(test_image)

print(Predictions[0])