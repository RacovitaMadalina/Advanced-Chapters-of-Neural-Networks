import keras
import numpy as np

from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Lambda
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

num_classes = 10
weight_decay = 1e-4

Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

model = Sequential()
model.add(Lambda(lambda x: x / 255, input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=X_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# Increase training dataset by using data augmentation

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train)

# Training stage of the classification model

batch_size = 64

opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy', \
              optimizer=opt_rms, \
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', \
                              factor=0.2, \
                              patience=5, \
                              min_lr=0.0003)

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), \
                    steps_per_epoch=X_train.shape[0] // batch_size, \
                    epochs=200, \
                    verbose=1, \
                    validation_data=(X_test, Y_test), \
                    callbacks=[reduce_lr])

# Serializing the model

model.save("../model.h5")

# Evaluation stage

scores = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100, scores[0]))

