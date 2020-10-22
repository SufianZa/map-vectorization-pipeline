import h5py
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from sklearn.model_selection import train_test_split
from os import listdir
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, \
    Conv2DTranspose, concatenate
import keras.backend as K
import keras.losses as L
import cv2
import os
import tensorflow_probability as tfp
from scipy.ndimage.filters import convolve
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from sklearn.utils import class_weight
# Load the TensorBoard notebook extension
import pickle
import tensorflow as tf
import datetime, os


class DeepDistanceTransform:
    def __init__(self):
        self.ddt = self.init_network((256, 256, 3))
        self.ddt.compile(loss='mean_squared_error', optimizer='adam', metrics=['RMSE', 'accuracy'])
        self.weight_file = "/content/drive/My Drive/Data/weights-checkpoint-{epoch:02d}.hdf5"

    def init_network(self, input_size):
        inputs = Input(input_size)
        c1, p1 = self.down_block(inputs, 16, drop=0.1)  # 128 -> 64
        c2, p2 = self.down_block(p1, 32, drop=0.1)  # 128 -> 64
        c3, p3 = self.down_block(p2, 64, drop=0.2)  # 64 -> 32
        c4, p4 = self.down_block(p3, 128, drop=0.2)  # 32 -> 16
        c5, p5 = self.down_block(p4, 256, drop=0.3)  # 16 -> 8
        cMid = self.bottleneck(p5, 512)
        u1 = self.up_block(cMid, c5, 256, strides=2, drop=0.3)  # 8 -> 16
        u2 = self.up_block(u1, c4, 128, strides=2, drop=0.3)  # 16 -> 32
        u3 = self.up_block(u2, c3, 64, strides=2, drop=0.3)  # 32 -> 64
        u4 = self.up_block(u3, c2, 32, strides=2, drop=0.3)  # 64 -> 128
        u5 = self.up_block(u4, c1, 16, strides=2, drop=0.3)  # 64 -> 128
        outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u5)
        return Model(inputs=[inputs], outputs=[outputs])

    def down_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1, drop=None):
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        p = MaxPooling2D((2, 2))(c)
        if drop: p = Dropout(drop)(p)
        return c, p

    def up_block(self, x, skip, filters, kernel_size=(3, 3), padding="same", strides=1, drop=None):
        up = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same')(x)
        concat = concatenate([up, skip])
        if drop: concat = Dropout(drop)(concat)
        c = Conv2D(filters, kernel_size, padding=padding, activation="relu")(concat)
        c = Conv2D(filters, kernel_size, padding=padding, activation="relu")(c)
        return c

    def bottleneck(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c

    def my_image_mask_generator(image_data_generator, mask_data_generator):
        train_generator = zip(image_data_generator, mask_data_generator)
        for (img, mask) in train_generator:
            yield (img, mask)

    def create_genrators(self, type):
        SEED = 100
        X_train = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1
        ).flow_from_directory('src/train/inputs', batch_size=16, target_size=(256, 256), seed=SEED)

        y_train = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1
        ).flow_from_directory('src/train/labels', batch_size=16, target_size=(256, 256), seed=SEED)
        return self.my_image_mask_generator(X_train, y_train)

    def train(self):
        checkpoint = ModelCheckpoint(self.weight_file, period=10, verbose=1, monitor='val_loss', save_best_only=True, mode='max')
        train_gen = self.create_genrators('train')
        val_gen = self.create_genrators('val')
        self.history = self.ddt.fit(train_gen,
                                    epochs=300,
                                    validation_data=val_gen,
                                    callbacks=[checkpoint])


if __name__ == '__main__':
    print('Train')
