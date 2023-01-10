import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds

max_len = 200
n_words = 10000
dim_embedding = 256
EPOCHS = 20
BATCH_SIZE = 500


def loadData():
    (X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data(num_words=n_words)

    X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

    return (X_train, Y_train), (X_test, Y_test)


def buildModel():
    model = models.Sequential()

    model.add(layers.Embedding(n_words, dim_embedding, input_length=max_len))
    model.add(layers.Dropout(0.3))

    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


(X_train, Y_train), (X_test, Y_test) = loadData()

model = buildModel()
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

score = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)

print('Test score:', score[0])
print('Test accuracy:', score[1])
