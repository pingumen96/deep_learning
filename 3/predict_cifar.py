import numpy as np
import scipy.misc
from tf.keras.models import model_from_json
from tf.keras.optimizers import SGD

model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# Load image
img_names = ['cat.jpg', 'dog.jpg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)), (2, 0, 1)).astype('float32') for img_name in img_names]
imgs = np.array(imgs) / 255

# train
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

# predict
predictions = model.predict_classes(imgs)
print(predictions)
