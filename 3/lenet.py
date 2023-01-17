import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers

EPOCHS = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam()
VALIDATION_SPLIT = 0.2

IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NB_CLASSES = 10


def build(input_shape, classes):
    model = models.Sequential()
    # CONV => RELU => POOL
    model.add(layers.Conv2D(20, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # CONV => RELU => POOL
    model.add(layers.Conv2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Flatten => RELU layers
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    # a softmax classifier
    model.add(layers.Dense(classes, activation='softmax'))
    return model


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# cast
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

# initialize the optimizer and model
model = build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
model.summary()

# use tensorboard
callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]

# fit
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=callbacks)

score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])
