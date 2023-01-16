import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# generate random data
np.random.seed(0)
area = 2.5 * np.random.randn(100) + 25.0
price = 25 * area + 5 + np.random.randint(20, 50, size=len(area))

# convert data to numpy array
data = np.array([area, price])
data = pd.DataFrame(data=data.T, columns=['area', 'price'])

# plot data
plt.scatter(data['area'], data['price'])
plt.show()

# calculate regression coefficients
W = sum(price * (area - np.mean(area))) / sum((area - np.mean(area)) ** 2)
b = np.mean(price) - W * np.mean(area)
print("The regression coefficients are: W = {}, b = {}".format(W, b))

y_pred = W * area + b

plt.plot(area, y_pred, color='red', label='Predicted price')
plt.scatter(data['area'], data['price'], label='Training data')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()