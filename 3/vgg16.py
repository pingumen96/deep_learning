import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def VGG_16(weights_path=None):
    model = models.Sequential()
    model.add(layers.ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


# read the text file imagenet1000_clsid_to_labels.txt
# the format of each line is: class_id: class_name
class_names = [line.strip().split(': ')[1] for line in open('imagenet1000_clsidx_to_labels.txt')]


def classToName(classNum):
    return class_names[classNum]


cat = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
#cat = im.transpose((2,0,1))
cat = np.expand_dims(cat, axis=0)

model = VGG_16('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy')
out = model.predict(cat)
print(classToName(np.argmax(out)))

ferrari = cv2.resize(cv2.imread('ferrari.jpg'), (224, 224)).astype(np.float32)
#ferrari = im.transpose((2,0,1))
ferrari = np.expand_dims(ferrari, axis=0)

model = VGG_16('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy')
out = model.predict(ferrari)
print(classToName(np.argmax(out)))
