import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the VGG16 model
model = VGG16(weights='imagenet', include_top=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy')

# Load the image and convert it into a numpy array
im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
im = np.expand_dims(im, axis=0)

# Make a prediction and get the index of the highest prediction
out = model.predict(im)
index = np.argmax(out)

# Print the index
print(index)

# Plot the output array
plt.plot(out.ravel())
plt.show()
