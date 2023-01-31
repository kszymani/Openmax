import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.datasets import fashion_mnist, mnist
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.layers import MaxPooling2D
from keras.layers import Softmax, GlobalAveragePooling2D, multiply, concatenate, MaxPool2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.backend import manual_variable_initialization
dataset = mnist
# dataset = fashion_mnist

batch_size = 256
epochs = 500


def Inception(_in, filters=None):
    if filters is None:
        filters = [10, 10, 10]
    col_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(_in)
    col_1 = Conv2D(filters[0], (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(col_1)
    col_2 = Conv2D(filters[1], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(_in)
    col_2 = Conv2D(filters[1], (5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(col_2)
    col_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
    col_3 = Conv2D(filters[2], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(col_3)
    out = concatenate([col_1, col_2, col_3])  # output size W x H x (f0 + f1 + f2)
    return out


def SqueezeExcite(_in, ratio=8):
    filters = _in.shape[-1]
    x = GlobalAveragePooling2D()(_in)
    x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
    return multiply([_in, x])


def mnist_model(input_shape, n_classes):
    _in = Input(shape=input_shape)
    x = Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(_in)
    x = SqueezeExcite(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), )(x)
    x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(x)
    x = SqueezeExcite(x)
    x = Inception(x, filters=[64, 64, 64])
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), )(x)
    x = Inception(x, filters=[16, 16, 16])
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), )(x)
    x = Flatten()(x)
    x = Dense(n_classes, kernel_initializer='he_normal')(x)
    x = Softmax()(x)
    model = Model(_in, x)
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    model.summary()
    return model


class PlotProgress(Callback):
    max_acc = 0
    max_val_acc = 0
    min_loss = sys.maxsize
    min_val_loss = sys.maxsize

    def on_train_begin(self, logs=None):
        plt.ion()
        self.first_epoch = True
        if logs is None:
            logs = {}
        self.metrics = {}
        self.f, self.axs = plt.subplots(1, 3, figsize=(15, 5))

        for metric in logs:
            self.metrics[metric] = []

    def on_train_end(self, logs=None):
        self.f.savefig("metrics")

    def on_epoch_end(self, epoch, logs=None):
        # Storing metrics
        if logs is None:
            logs = {}
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        self.max_acc = max(self.max_acc, logs.get("accuracy"))
        self.max_val_acc = max(self.max_val_acc, logs.get("val_accuracy"))
        self.min_loss = min(self.min_loss, logs.get("loss"))
        self.min_val_loss = min(self.min_val_loss, logs.get("val_loss"))

        metrics = [x for x in logs if 'val' not in x]
        for i, metric in enumerate(metrics):
            self.axs[i].plot(range(1, epoch + 2), self.metrics[metric], color='blue', label=metric)
            if 'val_' + metric in logs:
                self.axs[i].plot(range(1, epoch + 2), self.metrics['val_' + metric], label='val_' + metric,
                                 color='orange', )
                if metric == 'accuracy':
                    self.axs[i].set_title(
                        f"{'Max accuracy': <25}: {self.max_acc:.4f}\n{'Max val_accuracy': <25}: {self.max_val_acc:.4f}")
                elif metric == 'loss':
                    self.axs[i].set_title(
                        f"{'Min loss': <25}: {self.min_loss:.4f}\n{'Min val_loss': <25}: {self.min_val_loss:.4f}")
            if self.first_epoch:
                self.axs[i].legend()
                self.axs[i].grid()
        self.first_epoch = False
        plt.tight_layout()
        self.f.canvas.draw()
        self.f.canvas.flush_events()


def main():
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train = np.expand_dims(x_train, axis=-1)/255
    x_test = np.expand_dims(x_test, axis=-1)/255

    y_train, y_test = y_train.flatten(), y_test.flatten()
    classes = np.unique(y_train)
    print(classes)

    y_test = to_categorical(y_test, len(classes))
    y_train = to_categorical(y_train, len(classes))
    K = len(classes)

    print("number of classes:", K)
    print("input shape", x_train[0].shape)
    model = mnist_model(x_train[0].shape, K)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, shuffle=True)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.01,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    validgen = ImageDataGenerator()

    train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)
    valid_gen = validgen.flow(x_val, y_val, batch_size=batch_size)
    # plt.imshow(train_gen.next()[0][0], cmap='gray')
    # plt.show()
    history = model.fit(
        train_gen,
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=x_val.shape[0] // batch_size,
        callbacks=[
            EarlyStopping(monitor="val_loss", verbose=1, patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=8, verbose=1, min_lr=0.00001),
            PlotProgress(),
        ],
    )
    model.evaluate(x_train, y_train)
    model.evaluate(x_test, y_test)
    model.save("mnist")

    plt.plot(history.history['accuracy'], label='acc', color='red')
    plt.plot(history.history['val_accuracy'], label='val_acc', color='green')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
