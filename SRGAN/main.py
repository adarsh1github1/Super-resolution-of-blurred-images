import os

import warnings

warnings.filterwarnings("ignore")

import keras.backend as k
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Flatten, Layer
from tensorflow.keras.optimizers import Adam

tensorflow.config.run_functions_eagerly(True)


def get_lr():
    lr_images = []
    for i in glob.glob('../input/super-image-resolution/Data/LR/*'):
        img = cv2.imread(i)
        lr_images.append(img)
    return lr_images


def get_hr():
    hr_images = []
    for i in glob.glob('../input/super-image-resolution/Data/HR/*'):
        img = cv2.imread(i)
        hr_images.append(img)
    return hr_images


# lr size = 96 hr_size = 384

# def low_resolution(data,size,factor=2):
#     n_h = size / factor
#     n_w = size / factor
#     lr_images = []
#     for i in data:
#         img = cv2.imread(i)
#         img = cv2.resize(img,(n_h,n_w),interpolation=cv2.INTER_AREA)
#         #img = cv2.resize(img , (size,size), interpolation=cv2.INTER_AREA)
#         lr_images.append(img)
#     return lr_images

class loss_function(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def vgg_loss(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        for i in vgg19.layers:
            i.trainable = False
        model = tensorflow.keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return k.mean(k.square(model(y_true) - model(y_pred)))


def get_discriminator(ip_shape=(384, 384, 3)):
    loss = loss_function(ip_shape)

    input = tensorflow.keras.Input(shape=ip_shape)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(input)
    LR1 = LeakyReLU(alpha=0.2)(conv1)
    dp1 = Dropout(0.4)(LR1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(dp1)
    LR2 = LeakyReLU(alpha=0.2)(conv2)
    dp2 = Dropout(0.4)(LR2)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(dp2)
    LR3 = LeakyReLU(alpha=0.2)(conv3)
    dp3 = Dropout(0.2)(LR3)
    flatten = Flatten()(dp3)
    output = Dense(1, activation='sigmoid')(flatten)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)

    model = tensorflow.keras.Model(inputs=input, outputs=output)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_generator(ip_shape=(96, 96, 3)):
    input11 = tensorflow.keras.Input(shape=ip_shape)
    conv11 = Conv2D(64, kernel_size=(7, 7), padding='same')(input11)
    lr11 = LeakyReLU(alpha=0.2)(conv11)
    dp11 = Dropout(0.3)(lr11)
    conv22 = Conv2D(64, kernel_size=(3, 3), padding='same')(dp11)
    lr22 = LeakyReLU(alpha=0.2)(conv22)
    dp22 = Dropout(0.3)(lr22)
    conv33 = Conv2D(16, kernel_size=(3, 3), padding='same')(dp22)
    lr33 = LeakyReLU(alpha=0.2)(conv33)
    dp33 = Dropout(0.3)(lr33)
    conv44 = Conv2DTranspose(128, kernel_size=(4, 4), padding='same', strides=(2, 2))(dp33)
    lr44 = LeakyReLU(alpha=0.2)(conv44)
    ct11 = Conv2DTranspose(128, kernel_size=(4, 4), padding='same', strides=(2, 2))(lr44)
    lr55 = LeakyReLU(alpha=0.2)(ct11)
    output11 = Conv2D(3, kernel_size=(3, 3), activation='relu', padding='same')(lr55)

    model = Model(inputs=input11, outputs=output11)
    return model


def get_combined_gan(generator, discriminator, ip_shape=(384, 384, 3)):
    discriminator.trainable = False
    loss = loss_function(ip_shape)
    input_gan = tensorflow.keras.Input((96, 96, 3))
    generated = generator(input_gan)
    # print(generated.shape)
    gan_output = discriminator(generated)
    # print(gan_output.shape)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model = Model(inputs=input_gan, outputs=[gan_output, generated])
    model.compile(optimizer=opt, loss=['binary_crossentropy', loss.vgg_loss], loss_weights=[1e-3, 1.],
                  metrics=['accuracy'])

    return model


def save_plot(examples, epoch, n=3):
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis('off')
        plt.imshow(examples[i])
        filename = 'generated_plot_e%03d.png' % (epoch + 1)
        plt.savefig(filename)
        print("plots are getting printed")
        plt.close()


def get_real_samples(hr_images, n_samples):
    hr_images = np.array(hr_images)
    indexes = np.random.randint(0, len(hr_images), n_samples)
    x = hr_images[indexes]
    y = np.ones((n_samples, 1))
    return x, y


def get_fake_samples(lr_images, generator, n_samples):
    lr_images = np.array(lr_images)
    indexes = np.random.randint(0, len(lr_images), n_samples)
    lr = lr_images[indexes]
    x = generator.predict(lr)
    y = np.zeros((n_samples, 1))
    return x, y


def summary_so_far(hr_images, lr_images, epoch, generator, discriminator, n_samples):
    x_real, y_real = get_real_samples(hr_images, n_samples)
    _, real_accuracy = discriminator.evaluate(x_real, y_real, verbose=0)
    x_fake, y_fake = get_fake_samples(lr_images, generator, n_samples)
    _, fake_accuracy = discriminator.evaluate(x_fake, y_fake, verbose=0)

    print(">> Accuracy real: %.0f%%, fake: %.0f%% " % (real_accuracy * 100, fake_accuracy * 100))
    save_plot(x_fake, epoch)
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    generator.save(filename)


def train(lr_images, hr_images, generator, discriminator, epochs, n_samples):
    batch_per_epoch = int(len(lr_images) / n_samples)
    half_batch = int(n_samples / 2)
    for i in range(epochs):
        for j in range(batch_per_epoch):
            x_real, y_real = get_real_samples(hr_images, half_batch)
            x_fake, y_fake = get_fake_samples(lr_images, generator, half_batch)
            x, y = np.vstack((x_fake, x_real)), np.vstack((y_fake, y_real))
            d_loss = discriminator.train_on_batch(x, y)
            # print(d_loss)
            indexes = np.random.randint(0, len(lr_images), n_samples)
            lr_images = np.array(lr_images)
            hr_images = np.array(hr_images)
            trainx = lr_images[indexes]
            trainy = np.ones((n_samples, 1))
            trainy_images_vgg = hr_images[indexes]
            gan_model = get_combined_gan(generator, discriminator, ip_shape=(384, 384, 3))
            gan_loss = gan_model.train_on_batch(trainx, [trainy, trainy_images_vgg])

            # print(gan_loss)
            # print('>> Metrics are %d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, batch_per_epoch, d_loss))
            # print('>> Metrics are %d, %d/%d, d=%.3f' % (i + 1, j + 1, batch_per_epoch, d_loss))
        if i % 10 == 0:
            summary_so_far(hr_images, lr_images, epochs, generator, discriminator, n_samples)


hr_images = get_hr()
lr_images = get_lr()

generator = get_generator()
# generator.summary()
discriminator = get_discriminator()
# discriminator.summary()
epochs = 100
n_samples = 10

train(lr_images, hr_images, generator, discriminator, epochs, n_samples)
