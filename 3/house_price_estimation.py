import tensorflow as tf
import pandas as pd
import tensorflow.feature_column as fc
from tensorflow.keras.datasets import boston_housing

# load data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# convert data to pandas dataframe
x_train_data = pd.DataFrame(x_train, columns=features)
x_test_data = pd.DataFrame(x_test, columns=features)
y_train_data = pd.DataFrame(y_train, columns=['MEDV'])
y_test_data = pd.DataFrame(y_test, columns=['MEDV'])
x_train_data.head()

# create feature columns
feature_columns = []

for feature_name in features:
    feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

# create input functions


def estimator_input_fn(df_data, df_label, epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(df_data), df_label))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(epochs)
        return ds
    return input_function


train_input_fn = estimator_input_fn(x_train_data, y_train_data)
val_input_fn = estimator_input_fn(x_test_data, y_test_data, epochs=1, shuffle=False)

# create estimator
linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir='models/linear_regression')
linear_est.train(train_input_fn, steps=100)
result = linear_est.evaluate(val_input_fn)

# print results
result = linear_est.predict(val_input_fn)
for pred, exp in zip(result, y_test[:32]):
    print('Predicted value: {}, Expected value: {}'.format(pred['predictions'][0], exp))
