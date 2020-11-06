"""
This class is Distance Transformation Estimator to predict the Euclidian distance transform of each map tile
"""

from PIL import Image
import numpy as np
from pathlib import Path, PurePath
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, \
    Conv2DTranspose, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import datetime, os
import global_variables


def addPadding(image, top=0, bottom=0, left=0, right=0, a=0):
    image = cv2.copyMakeBorder(image.copy(), top + a, bottom + a, left + a, right + a, cv2.BORDER_CONSTANT, 255)
    image[image == 0] = 255
    return image


def removePadding(image, top=0, bottom=0, left=0, right=0, a=0):
    w, h = image.shape
    return image.copy()[left + a:w - bottom - a, top + a:h - right - a]


class DeepDistanceTransform:
    def __init__(self, batch_size=64, epochs=30, window_size=256):
        self.batch_size = batch_size
        self.window_size = window_size
        self.epochs = epochs
        self.weight_file = str(Path(global_variables.weights_path, 'best_weight.hdf5'))
        self.model = self.init_network((window_size, window_size, 3))
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model.load_weights(self.weight_file)

    def init_network(self, input_size):
        """
        This method initiates the u-net architecture of Deep Distance Transform
        which takes the input_size as initial size of the Input layer
        """
        inputs = Input(input_size)
        skip1, pool1 = self.encode(inputs, 16, drop=0.1)
        skip2, pool2 = self.encode(pool1, 32, drop=0.1)
        skip3, pool3 = self.encode(pool2, 64, drop=0.2)
        skip4, pool4 = self.encode(pool3, 128, drop=0.2)
        skip5, pool5 = self.encode(pool4, 256, drop=0.3)
        bottleneck = self.bottleneck(pool5, 512)
        deConv1 = self.decode(bottleneck, skip5, 256, strides=2, drop=0.3)
        deConv2 = self.decode(deConv1, skip4, 128, strides=2, drop=0.3)
        deConv3 = self.decode(deConv2, skip3, 64, strides=2, drop=0.3)
        deConv4 = self.decode(deConv3, skip2, 32, strides=2, drop=0.2)
        deConv5 = self.decode(deConv4, skip1, 16, strides=2, drop=0.1)
        outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(deConv5)
        return Model(inputs=inputs, outputs=[outputs])

    def encode(self, x, filters, kernel_size=(3, 3), padding="same", strides=1, drop=None):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(conv)
        pool = MaxPooling2D((2, 2))(conv)
        if drop: pool = Dropout(drop)(pool)
        return conv, pool

    def bottleneck(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(conv)
        return conv

    def decode(self, x, skip, filters, kernel_size=(3, 3), padding="same", strides=1, drop=None):
        deConv = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same')(x)
        concat = concatenate([deConv, skip])
        if drop: concat = Dropout(drop)(concat)
        deConv = Conv2D(filters, kernel_size, padding=padding, activation="relu")(concat)
        deConv = Conv2D(filters, kernel_size, padding=padding, activation="relu")(deConv)
        return deConv

    def input_label_generator(self, mode='train'):
        """
        This method provides pairs of input images and labels as generators
        and can be used for train as well as validation data
        :param mode: str
                can be set for "train" or "val" data
                and should match the dataset path in global variables
        """
        SEED = 700
        data_gen_args = dict(rescale=1. / 255,
                             width_shift_range=0.4,
                             height_shift_range=0.4,
                             fill_mode='wrap',
                             zoom_range=0.5,
                             horizontal_flip=True,
                             vertical_flip=True)

        X_train = ImageDataGenerator(**data_gen_args).flow_from_directory(
            str(global_variables.train_test_val_path[mode]['x'].parent), batch_size=self.batch_size, color_mode='rgb',
            seed=SEED)
        y_train = ImageDataGenerator(**data_gen_args).flow_from_directory(
            str(global_variables.train_test_val_path[mode]['y'].parent), batch_size=self.batch_size,
            class_mode='input', color_mode='grayscale',
            seed=SEED)
        while True:
            yield next(X_train)[0], next(y_train)[0]

    def train(self):
        checkpoint = ModelCheckpoint(self.weight_file, verbose=1, monitor='val_loss', save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=10,
                                   verbose=0, mode='auto')

        train_gen = self.input_label_generator('train')
        val_gen = self.input_label_generator('val')

        _, _, num_of_train = next(os.walk(str(global_variables.train_test_val_path['train']['x'])))
        _, _, num_of_val = next(os.walk(str(global_variables.train_test_val_path['val']['x'])))

        self.history = self.model.fit(train_gen,
                                      steps_per_epoch=len(num_of_train) // self.batch_size,
                                      epochs=25,
                                      validation_steps=len(num_of_val) // self.batch_size,
                                      validation_data=val_gen,
                                      callbacks=[checkpoint, early_stop])

    def test(self):
        """
            Tests the model on the test images in the pre-defined paths in global variables
            then plots a comparision of the prediction and ground truth patches
        """
        x = []
        y = []
        for img_path in global_variables.train_test_val_path['test']['x'].glob('*.*'):
            name = os.path.basename(img_path)
            img = np.array(Image.open(img_path))
            mask = np.array(
                Image.open(os.path.join(global_variables.train_test_val_path['test']['y'], name)))  # read image
            mask = (mask - np.amin(mask)) * 1.0 / (np.amax(mask) - np.amin(mask))
            img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))
            x.append(img)
            y.append(mask)
        x = np.array(x)
        y = np.array(y)
        output = self.model.predict(x, verbose=0)
        for i in range(output.shape[0]):
            inp = np.squeeze(x[i, ...])
            pre = np.squeeze(output[i, ...])
            ori = np.squeeze(y[i, ...])
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle('Deep Distance Transform Estimation {}'.format(i))
            ax1.imshow(inp)
            ax2.imshow(pre, cmap='gray')
            ax3.imshow(ori, cmap='gray')

            ax1.title.set_text('input')
            ax2.title.set_text('Estimated')
            ax3.title.set_text('Ground truth')
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            plt.show()

    def estimate_full_map(self, input_map, trim=60):
        """
         Estimates the full map image by sliding a window over and
           trimming off sides from each side of 256*256 batch
           e.g. 100 -> adds only the middle 56*56 square of the 256*256 batch to the result.
           The trimming is used to avoid creases and artifacts that affect the watershed segmentation
        :param input_map: ndarray
            image of cadastral map
        :param trim: int
            the number of pixels trimmed of each side of the predicted window
        """
        w, h, _ = input_map.shape
        stepSize = self.window_size - trim * 2
        pad_bottom = self.window_size - (input_map.shape[0] % self.window_size)
        pad_right = self.window_size - (input_map.shape[1] % self.window_size)
        input_map = addPadding(input_map, bottom=pad_bottom, right=pad_right, a=trim * 2)
        input_map = (input_map - np.amin(input_map)) * 1.0 / (np.amax(input_map) - np.amin(input_map))
        in_image = np.reshape(input_map, (1, input_map.shape[0], input_map.shape[1], input_map.shape[2]))
        res = np.zeros((input_map.shape[0], input_map.shape[1]))
        for y in range(0, input_map.shape[1], stepSize):
            for x in range(0, input_map.shape[0], stepSize):
                window = in_image[:, x:x + self.window_size, y:y + self.window_size, :]
                if window.shape[1] == self.window_size and window.shape[2] == self.window_size:
                    output = self.model.predict(window, verbose=0)
                    res[x + trim:x + self.window_size - trim,
                    y + trim:y + self.window_size - trim] = output.squeeze()[
                                                            trim:self.window_size - trim,
                                                            trim:self.window_size - trim]
        res = removePadding(res, bottom=pad_bottom, right=pad_right, a=trim * 2)
        assert res.shape[0] == w and res.shape[1] == h
        print(res.shape[0], res.shape[1])

        plt.imshow(res, cmap='gray')
        plt.show()

        out_im = Image.fromarray(res)
        return np.array(out_im)


if __name__ == '__main__':
    model = DeepDistanceTransform(batch_size=24, epochs=50)
    image = np.array(Image.open(Path(global_variables.test_full_maps, '5.tif')))[:, :, :3]
    model.estimate_full_map(image, trim=50)
