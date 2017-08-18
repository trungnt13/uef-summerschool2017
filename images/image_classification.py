# ===========================================================================
# NOTE: the performance of float16 and float32 dataset are identical
# ===========================================================================
from __future__ import print_function, absolute_import, division

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import os
import sys

from scipy.stats import itemfreq
import numpy as np
np.random.seed(1208)

from utils import (rotate, shift, zoom, shear, one_hot, cifar_labels,
                   plot_hist)

from keras.utils.generic_utils import Progbar
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop

# ===========================================================================
# Helper and constants
# ===========================================================================
NB_OF_VISUALIZATION = 25
TRAINING_PORTION = 0.5
VALID_PORTION = 0.2

current_script_path = os.path.dirname(sys.argv[0])
current_script_path = os.path.join('.', current_script_path)

MNIST_x_path = os.path.join(current_script_path, 'MNIST_x.npy')
MNIST_y_path = os.path.join(current_script_path, 'MNIST_y.npy')
print("MNIST path:", MNIST_x_path, MNIST_y_path)

CIFAR_x_path = os.path.join(current_script_path, 'CIFAR_x.npy')
CIFAR_y_path = os.path.join(current_script_path, 'CIFAR_y.npy')
print("CIFAR path:", CIFAR_x_path, CIFAR_y_path)


def train_and_test_dnn(x_train, y_train,
                       x_valid, y_valid,
                       x_test, y_test,
                       title):
    input_shape = x_train.shape[1:]
    model = Sequential()
    model.add(Reshape(target_shape=(np.prod(input_shape),),
                      input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    #showing the network configuration
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=5,
                        verbose=1,
                        validation_data=(x_valid, y_valid))
    # ====== plot history ====== #
    plt.figure()
    plt.subplot(1, 2, 1)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss, color='b', label='Training loss')
    plt.plot(val_loss, color='r', label="Validing loss")
    plt.suptitle(title + "(cross-entropy loss)")
    plt.legend()
    plt.subplot(1, 2, 2)
    train_loss = history.history['acc']
    val_loss = history.history['val_acc']
    plt.plot(train_loss, color='b', label='Training Accuracy')
    plt.plot(val_loss, color='r', label="Validing Accracy")
    plt.suptitle(title + "(Accuracy)")
    plt.legend()
    # ====== final evaluation ====== #
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history


def train_and_test_cnn(x_train, y_train,
                       x_valid, y_valid,
                       x_test, y_test,
                       title):
    model = Sequential()
    model.add(Conv2D(32, 3, 3,
                     input_shape=x_train.shape[1:],
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 3, 3,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=5,
                        verbose=1,
                        validation_data=(x_valid, y_valid))
    # ====== plot history ====== #
    plt.figure()
    plt.subplot(1, 2, 1)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss, color='b', label='Training loss')
    plt.plot(val_loss, color='r', label="Validing loss")
    plt.suptitle(title + "(cross-entropy loss)")
    plt.legend()
    plt.subplot(1, 2, 2)
    train_loss = history.history['acc']
    val_loss = history.history['val_acc']
    plt.plot(train_loss, color='b', label='Training Accuracy')
    plt.plot(val_loss, color='r', label="Validing Accracy")
    plt.suptitle(title + "(Accuracy)")
    plt.legend()
    # ====== final evaluation ====== #
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history

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
    ax.set_title("Number: %d" % y[i])
plt.tight_layout()

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

nb_train = int(TRAINING_PORTION * len(x))
nb_valid = int(VALID_PORTION * len(x))
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
plot_hist(y_train, ax, "Training distribution")
ax = plt.subplot(3, 1, 2)
plot_hist(y_train, ax, "Validating distribution")
ax = plt.subplot(3, 1, 3)
plot_hist(y_train, ax, "Testing distribution")
plt.tight_layout()

# ====== convert labels to one_hot for training ====== #
labels = ["Number: %d" % i for i in y_train[:16]]
y_train = one_hot(y_train, nb_classes=10)
y_aug = one_hot(y_aug, nb_classes=10)
y_test = one_hot(y_test, nb_classes=10)
y_valid = one_hot(y_valid, nb_classes=10)
plt.figure()
plt.imshow(y_train[:16], cmap=plt.cm.Greys_r)
plt.xticks(np.arange(10))
plt.yticks(np.arange(16), labels)
plt.suptitle("One-hot labels matrix")

# ====== show everything ====== #
plt.show()

# ====== create and training the network ====== #
print("********************************")
print("[MNIST] On original training set")
print("********************************")
train_and_test_dnn(x_train, y_train, x_valid, y_valid, x_test, y_test,
    'MNIST Original data')

# ====== create and training the network on Augmented data ====== #
print("********************************")
print("[MNIST] On augmented training set")
print("********************************")
train_and_test_dnn(x_aug, y_aug, x_valid, y_valid, x_test, y_test,
    'MNIST Augmented data')

# ===========================================================================
# Working on colored images dataset (CIFAR)
# ===========================================================================
x = np.load(CIFAR_x_path)
y = np.load(CIFAR_y_path)
print("CIFAR shapes:", x.shape, y.shape)

# ====== visualize the data ====== #
fig = plt.figure()
labels = ''
n = int(np.sqrt(NB_OF_VISUALIZATION))
for i in range(NB_OF_VISUALIZATION):
    ax = plt.subplot(n, n, i + 1)
    ax.axis('off')
    # plot grey scale image require 2D array
    plt.imshow(x[i])
    ax.set_title(cifar_labels[y[i]], fontsize=10)

# ====== plot differnt channel ====== #
fig = plt.figure()
sample_img = x[8]
title = ['R', 'G', 'B']
for i in range(3):
    temp = np.zeros(sample_img.shape, dtype='uint8')
    temp[:, :, i] = sample_img[:, :, i]
    ax = plt.subplot(1, 3, i + 1)
    ax.imshow(temp)
    ax.set_axis_off()
    ax.set_title("Channel: " + title[i])

# ====== again split train, test, valid ====== #
# shuffle the data, note the order of x and y must match
permutation = np.random.permutation(len(x))
x = x[permutation]
y = y[permutation]

nb_train = int(TRAINING_PORTION * len(x))
nb_valid = int(VALID_PORTION * len(x))
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
print("Augmented set:", x_aug.shape, y_aug.shape)
print("Valid set:", x_valid.shape, y_valid.shape)
print("Test set:", x_test.shape, y_test.shape)
print("Augmented training set:", x_aug.shape, y_aug.shape)

# ====== convert labels to one_hot for training ====== #
y_train = one_hot(y_train, nb_classes=10)
y_aug = one_hot(y_aug, nb_classes=10)
y_test = one_hot(y_test, nb_classes=10)
y_valid = one_hot(y_valid, nb_classes=10)

# ====== train FNN on CIFAR ====== #
print("************************************")
print("[CIFAR] On unnormalized training set")
print("************************************")
train_and_test_dnn(x_train, y_train, x_valid, y_valid, x_test, y_test,
    "CIFAR-DNN unnormalized data")

# ====== fail to converge, normalize the data ====== #
x_train = x_train.astype('float32')
x_train /= 255.
x_aug = x_aug.astype('float32')
x_aug /= 255.
x_valid = x_valid.astype('float32')
x_valid /= 255.
x_test = x_test.astype('float32')
x_test /= 255.

# ====== Train one more time with normalized data ====== #
print("************************************")
print("[CIFAR] On NORMALIZED training set")
print("************************************")
train_and_test_dnn(x_train, y_train, x_valid, y_valid, x_test, y_test,
    "CIFAR-DNN normalized data")

# ====== Using CNN to improve ====== #
print("************************************")
print("[CIFAR] CNN original training set")
print("************************************")
train_and_test_cnn(x_train, y_train, x_valid, y_valid, x_test, y_test,
    "CIFAR-CNN normalized data")

# ====== Using CNN with augmented data ====== #
print("************************************")
print("[CIFAR] CNN Augmented training set")
print("************************************")
train_and_test_cnn(x_aug, y_aug, x_valid, y_valid, x_test, y_test,
    "CIFAR_CNN augmented data")

from odin.visual import plot_save
plot_save("/tmp/tmp.pdf")
