import os
import time
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# Questo codice utilizza il framework di deep learning Tensorflow per addestrare un modello di classificazione immagine per
# riconoscere cavalli o esseri umani. Utilizza il dataset "horses_or_humans" di Tensorflow_datasets per addestrare e validare il modello.
#
# In primo luogo, il codice carica il dataset "horses_or_humans" e lo suddivide in tre gruppi: train, validation e test.
# Successivamente, mostra alcune immagini casuali dal dataset di addestramento utilizzando la funzione "show_images".
#
# Successivamente, il codice formatta le immagini del dataset in modo che abbiano tutte la stessa dimensione di 160x160 pixel e le normalizza per avere valori tra - 1 e 1.
#
# Il codice utilizza quindi il modello MobileNetV2 pre-addestrato per estrarre caratteristiche dalle immagini del dataset.
# MobileNetV2 è un modello di computer vision pre-addestrato su un grande dataset di immagini chiamato ImageNet.
#
# Successivamente, il codice utilizza una GlobalAveragePooling2D per ridurre le dimensioni delle caratteristiche estratte dalle immagini e quindi utilizza una fully connected layer per generare una predizione.
#
# Infine, il codice utilizza un ottimizzatore RMSprop per addestrare il modello e lo valida utilizzando il dataset di validazione. Il codice stampa anche alcune informazioni sulla forma delle immagini, delle caratteristiche #estratte e delle predizioni generate durante il processo di addestramento.
SPLIT_WEIGHTS = (8, 1, 1)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'horses_or_humans', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True, as_supervised=True)

get_label_name = metadata.features['label'].int2str


def show_images(dataset):
    for image, label in dataset.take(10):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))


show_images(raw_train)

IMG_SIZE = 160


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 2000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
base_model.summary()

for image_batch, label_batch in train_batches.take(1):
    pass
print(image_batch.shape)

feature_batch = base_model(image_batch)
print(feature_batch.shape)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

num_train, num_val, num_test = (
    metadata.splits['train'].num_examples * weight / 10
    for weight in SPLIT_WEIGHTS
)

initial_epochs = 10
steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = 4

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
