import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Load the data
((train_data, train_labels), (eval_data, eval_labels)) = keras.datasets.mnist.load_data()

# Normalize the data
train_data = train_data / np.float32(255)
train_labels = train_labels.astype(np.int32)

eval_data = eval_data / np.float32(255)
eval_labels = eval_labels.astype(np.int32)

feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]
classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=10, model_dir="/tmp/mnist_model")

# Define the training input
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

# Train the model
classifier.train(input_fn=train_input_fn, steps=10000)

# Define the evaluation input
val_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

# Evaluate the model
eval_results = classifier.evaluate(input_fn=val_input_fn)
print(eval_results)
