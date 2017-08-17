# ===========================================================================
# NOTE: the performance of float16 and float32 dataset are identical
# ===========================================================================
from __future__ import print_function, absolute_import, division

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import os
import sys

from scipy.stats import itemfreq
import numpy as np
np.random.seed(1208)

from utils import rotate, shift, zoom, shear, one_hot

from keras.utils import Progbar
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.optimizers import RMSprop

# ===========================================================================
# Helper and constants
# ===========================================================================
NB_OF_VISUALIZATION = 25

current_script_path = os.path.dirname(sys.argv[0])
current_script_path = os.path.join('.', current_script_path)

MNIST_x_path = os.path.join(current_script_path, 'MNIST_x.npy')
MNIST_y_path = os.path.join(current_script_path, 'MNIST_y.npy')
print("MNIST path:", MNIST_x_path, MNIST_y_path)

CIFAR_x_path = os.path.join(current_script_path, 'CIFAR_x.npy')
CIFAR_y_path = os.path.join(current_script_path, 'CIFAR_y.npy')
print("CIFAR path:", CIFAR_x_path, CIFAR_y_path)


# ===========================================================================
# Load MNIST dataset
# ===========================================================================
x = np.load(MNIST_x_path)
y = np.load(MNIST_y_path)
print("MNIST shapes:", x.shape, y.shape)

# ====== visualize the data ====== #
fig = plt.figure()
labels = ''
n = int(np.sqrt(NB_OF_VISUALIZATION))
for i in range(NB_OF_VISUALIZATION):
    ax = plt.subplot(n, n, i + 1)
    ax.axis('off')
    # plot grey scale image require 2D array
    plt.imshow(x[i][:, :, 0], cmap=plt.cm.Greys_r)
    # add labels
    if ((i + 1) % n) == 0:
        labels += ' ' + str(y[i]) + ' |'
    else:
        labels += ' ' + str(y[i])
plt.suptitle("Labels:" + labels)

# ====== augmentation the data ====== #
img = x[0]

fig = plt.figure()
ax = plt.subplot(2, 4, 1)
ax.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
ax.axis('off')
ax.set_title('Original image')

ax = plt.subplot(2, 4, 2)
ax.imshow(rotate(img, 45)[:, :, 0], cmap=plt.cm.Greys_r)
ax.axis('off')
ax.set_title('Rotated positive')

ax = plt.subplot(2, 4, 2)
ax.imshow(rotate(img, -45)[:, :, 0], cmap=plt.cm.Greys_r)
ax.axis('off')
ax.set_title('Rotated negative')

ax = plt.subplot(2, 4, 3)
ax.imshow(shift(img, 0.2, 0.2)[:, :, 0], cmap=plt.cm.Greys_r)
ax.axis('off')
ax.set_title('Shift positive')

ax = plt.subplot(2, 4, 4)
ax.imshow(shift(img, -0.2, -0.2)[:, :, 0], cmap=plt.cm.Greys_r)
ax.axis('off')
ax.set_title('Shift negative')

ax = plt.subplot(2, 4, 5)
ax.imshow(zoom(img, 2, 2)[:, :, 0], cmap=plt.cm.Greys_r)
ax.axis('off')
ax.set_title('Zoom small')

ax = plt.subplot(2, 4, 6)
ax.imshow(zoom(img, 0.8, 0.8)[:, :, 0], cmap=plt.cm.Greys_r)
ax.axis('off')
ax.set_title('Zoom big')

ax = plt.subplot(2, 4, 7)
ax.imshow(shear(img, 0.8)[:, :, 0], cmap=plt.cm.Greys_r)
ax.axis('off')
ax.set_title('Shear negative')

ax = plt.subplot(2, 4, 8)
ax.imshow(shear(img, -0.8)[:, :, 0], cmap=plt.cm.Greys_r)
ax.axis('off')
ax.set_title('Shear positive')

# ====== splitting train, valid, test ====== #
# shuffle the data, note the order of x and y must match
permutation = np.random.permutation(len(x))
x = x[permutation]
y = y[permutation]

nb_train = int(0.6 * len(x))
nb_valid = int(0.2 * len(x))
nb_test = len(x) - nb_train - nb_valid

x_train = x[:nb_train]
y_train = y[:nb_train]

x_valid = x[nb_train:nb_train + nb_valid]
y_valid = y[nb_train:nb_train + nb_valid]

x_test = x[nb_train + nb_valid:]
y_test = y[nb_train + nb_valid:]

# ====== augmenting the training data ====== #
augment_function = [lambda img: shift(img, 0.1, -0.2),
                    lambda img: rotate(img, 45)]
# apply on out data
x_new = []
y_new = []
prog = Progbar(target=len(x_train))
for i in range(len(x_train)):
    x_new += [x_train[i]] + [f(x_train[i]) for f in augment_function]
    y_new += [y_train[i]] * (1 + len(augment_function))
    prog.update(i)
prog.update(len(x_train))
x_aug = np.array(x_new)
y_aug = np.array(y_new)

# ====== print info ====== #
print("Train set:", x_train.shape, y_train.shape)
print("Valid set:", x_valid.shape, y_valid.shape)
print("Test set:", x_test.shape, y_test.shape)
print("Augmented training set:", x_aug.shape, y_aug.shape)

# ====== checking distribution of train, valid, test matching ====== #
train_dist = itemfreq(y_train)
valid_dist = itemfreq(y_valid)
test_dist = itemfreq(y_test)

plt.figure()
ax = plt.subplot(3, 1, 1)
ax.hist(y_train, bins=10)
ax.set_title("Training distribution")

ax = plt.subplot(3, 1, 2)
ax.hist(y_valid, bins=10)
ax.set_title("Validation distribution")

ax = plt.subplot(3, 1, 3)
ax.hist(y_test, bins=10)
ax.set_title("Testing distribution")
plt.tight_layout()

plt.show()

# ====== convert labels to one_hot for training ====== #
y_train = one_hot(y_train, nb_classes=10)
y_test = one_hot(y_test, nb_classes=10)
y_valid = one_hot(y_valid, nb_classes=10)
print("Onehot labels:")
print(y_train[:8])

# ====== create and training the network ====== #
model = Sequential()
model.add(Reshape(target_shape=(784,), input_shape=(28, 28, 1)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=8,
                    verbose=1,
                    validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
