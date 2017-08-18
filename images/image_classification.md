# Image processing and classification
### Tutors: [Juha Mehtonen, Trung Ngo Trong]()

-----

## Loading libraries and dataset

```python
%matplotlib inline
from __future__ import print_function, absolute_import, division
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
figsize(12, 6)

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

```

What do we need for this task:

* **numpy and scipy**: for loading images and matrix manipulation
* **utils**: for some shortcuts to images transformation
* **keras**: powerful neural network library for training the classifier

## Constants

Setting the amount of training, validating, and testing data here

```python
NB_OF_VISUALIZATION = 25
TRAINING_PORTION = 0.5
VALID_PORTION = 0.2
```


## Loading and visualize the MNIST dataset

```python
x = np.load('MNIST_x.npy')
y = np.load('MNIST_y.npy')
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
```

What can we see from the images?

* The image is grayscale, binary photos(i.e. features are 0 or 1)
* Not much rotation or translation appeared to different images from the same digit
* The labels are matched with the images, hence, the provided dataset is clean and sound

## Augmentation images for training

Since only _50 % _ of the dataset is used for training, it is good idea to perform data augmentation ?

The idea of data augmentation is:

>> > Increasing the amount of data by introducing noise, or transformation to original images to create a more robust dataset.

There are four basic image transformation can be applied:

* Rotation
* Translation(shift)
* Zooming
* Shearing

All are visualized by following figures.

```python
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
```

## Data splitting for training classifiers

We are interested in the generalization ability of the model to new data, it is important to create reliable datasets for evaluating this criterion.

Since the model is fitted on training set, the performance on training data is trivial.
As a result, we split the dataset into 3 partitions:

* Training set
* Validating set: for model selection, hyper - parameters fine tuning.
* Testing set: for final evaluation of the model, since the model has never seen those data, its performance is closest to the generalization ability

```python
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
```

Another important note is that we only perform augmentation on training data, followed by this code:

```python
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
```

It is critical to validate our splitting strategy.
The algorithm must assure that the splitting process doesn't create any bias in training dataset, which can be checked by visualizing the distribution of the true dataset.

```python
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
```

## Training digit classifier

Most of machine learning algorithm require the labels to be one - hot encoded, which is visualized by following code:

```python
# ====== convert labels to one_hot for training ====== #
labels = ["Number: %d" % i for i in y_train[:16]]
y_train = one_hot(y_train, nb_classes=10)
y_aug = one_hot(y_aug, nb_classes=10)
y_test = one_hot(y_test, nb_classes=10)
y_valid = one_hot(y_valid, nb_classes=10)
plt.figure()
plt.imshow(y_train[:16], cmap=plt.cm.Greys_r, interpolation='nearest')
plt.xticks(np.arange(10))
plt.yticks(np.arange(16), labels)
plt.suptitle("One-hot labels matrix")
```

### Creating neural network classifier with keras

The following function create a neural network, which output 10 probability values(i.e. softmax output) to represent the confident of given image for 10 different digits.

The network is trained on ** (x_train, y_train)**, validated on ** (x_valid, y_valid)**, and finally evaluated using ** (x_test, y_test)**

```python


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

    history = model.fit(x_train, y_train, 128, 5,
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
```

We first train the model on original MNIST training set:

```python
train_and_test_dnn(x_train, y_train, x_valid, y_valid, x_test, y_test,
    'MNIST Original data')
```

The model summary and learning curves are also procedure, we can use this information for diagnose the training process.

Let see if the augmented dataset really help in this case:

```python
train_and_test_dnn(x_aug, y_aug, x_valid, y_valid, x_test, y_test,
    'MNIST Augmented data')
```

Why the augmented data doesn't work out?

* The data is simple and there are no complicated transformations.
* The amount of training data is significantly increased, hence, it probably require more powerful network to learn additional representation.


## What about colored images?

We use[CIFAR - 10](http: // www.cs.toronto.edu / ~kriz / cifar.html) dataset for simulating similar approach to colored images.

The original CIFAR - 10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The data in this example is 15000 randomly selected images from CIFAR - 10.
The ten categories are:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Load and visualize the dataset:

```python
x = np.load('CIFAR_x.npy')
y = np.load('CIFAR_y.npy')
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
```

### Data preparation and splitting

We repeat the same process applied for MNIST

```python
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
```

### Training classifier

Again, we apply the same network for CIFAR dataset:

```python
# ====== train FNN on CIFAR ====== #
train_and_test_dnn(x_train, y_train, x_valid, y_valid, x_test, y_test,
    "CIFAR-DNN unnormalized data")
```

The training is not converged, there is no improvement in the validation set as well as the training.
The test accuracy is about _8%_ which is below random guess.
So what **wrong**?

### Normalizing the data

It is notable that the CIFAR-10 provided features in **uint8** data type, and the intensity of each pixel is from 0 to 255.
The big values would magnify the backpropagated gradients values, as a results, the weights are moving around so fast that they cannot reach a better solution.

We normalize the values in range **[0., 1.]**. Then training the network again

```python
# ====== fail to converge, normalize the data ====== #
x_train = x_train.astype('float32')
x_train /= 255.
x_aug = x_aug.astype('float32')
x_aug /= 255.
x_valid = x_valid.astype('float32')
x_valid /= 255.
x_test = x_test.astype('float32')
x_test /= 255.
train_and_test_dnn(x_train, y_train, x_valid, y_valid, x_test, y_test,
    "CIFAR-DNN normalized data")
```

The network converged, and we get significant improvement, but it is nothing compare to _>90%_ on MNIST dataset.
Could we get a better model for this task?

### Convolutional Neural Network for image recognition

Creating convolutional neural network in keras is straight forward:

```python
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
    history = model.fit(x_train, y_train, 128, 5,
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
```

We again train the network on normalized CIFAR-10, but we also try it on augmented CIFAR-10

```python
train_and_test_cnn(x_train, y_train, x_valid, y_valid, x_test, y_test,
    "CIFAR-CNN normalized data")
```
Big improvement! Note our CNN network only need half of the amount of the parameters (~400,000) compared to fully connected network (~800,000), it demonstrate the efficiency of this network in learning image presentation.

```python
train_and_test_cnn(x_aug, y_aug, x_valid, y_valid, x_test, y_test,
    "CIFAR_CNN augmented data")
```
Event bigger improvement! Augmentation really work in this case