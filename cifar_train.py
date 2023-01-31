import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Input, BatchNormalization, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Softmax, GlobalAveragePooling2D, multiply, concatenate, MaxPool2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# dataset = fashion_mnist

batch_size = 128
epochs = 500

weight_decay = 1e-4


def Inception(_in, filters=None):
    if filters is None:
        filters = [10, 10, 10]
    col_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(_in)
    col_1 = Conv2D(filters[0], (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(col_1)

    col_2 = Conv2D(filters[1], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(_in)
    col_2 = Conv2D(filters[1], (5, 5), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(col_2)

    col_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
    col_3 = Conv2D(filters[2], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(col_3)

    out = concatenate([col_1, col_2, col_3])  # output size W x H x (f0 + f1 + f2)
    return out


def SqueezeExcite(_in, ratio=8):
    """Squeeze-and-Excitation layers are considered to improve CNN performance.
    `Find out more <https://doi.org/10.48550/arXiv.1709.01507>`
    """
    filters = _in.shape[-1]
    x = GlobalAveragePooling2D()(_in)
    x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    return multiply([_in, x])


def ReduceChannels(_in, channels=0):
    return Conv2D(channels, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=l2(weight_decay))(_in)


def cifar_model(input_shape, n_classes):
    _in = Input(shape=input_shape)
    d_r = 0.25
    x = Inception(_in, filters=[32, 32, 32])
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)  # 16

    x = Dropout(d_r)(x)
    x = Inception(x, filters=[64, 64, 64])
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)  # 8

    x = Dropout(d_r)(x)
    x = Inception(x, filters=[128, 128, 128])
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)  # 4

    x = GlobalAveragePooling2D()(x)
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

    acc_ep = 0
    val_acc_ep = 0
    loss_ep = 0
    val_loss_ep = 0

    def __init__(self, i_dir):
        super().__init__()
        self.axs = None
        self.f = None
        self.metrics = None
        self.i_dir = i_dir
        self.first_epoch = True

    def on_train_begin(self, logs=None):
        plt.ion()
        if logs is None:
            logs = {}
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
        self.f, self.axs = plt.subplots(1, 3, figsize=(13, 4))

    def on_train_end(self, logs=None):
        self.f.savefig(f"{self.i_dir}/metrics")

    def on_epoch_end(self, epoch, logs=None):
        # Storing metrics
        if logs is None:
            logs = {}
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        acc = max(self.max_acc, round(logs.get("accuracy"), 4))
        val_acc = max(self.max_val_acc, round(logs.get("val_accuracy"), 4))
        loss = min(self.min_loss, round(logs.get("loss"), 4))
        val_loss = min(self.min_val_loss, round(logs.get("val_loss"), 4))

        if acc == self.max_acc:
            self.acc_ep += 1
        else:
            self.acc_ep = 0
        if val_acc == self.max_val_acc:
            self.val_acc_ep += 1
        else:
            self.val_acc_ep = 0

        if loss == self.min_loss:
            self.loss_ep += 1
        else:
            self.loss_ep = 0

        if val_loss == self.min_val_loss:
            self.val_loss_ep += 1
        else:
            self.val_loss_ep = 0

        self.max_acc = acc
        self.max_val_acc = val_acc
        self.min_loss = loss
        self.min_val_loss = val_loss

        metrics = [x for x in logs if 'val' not in x]
        for i, metric in enumerate(metrics):
            self.axs[i].plot(range(1, epoch + 2), self.metrics[metric], color='blue', label=metric)
            if 'val_' + metric in logs:
                self.axs[i].plot(range(1, epoch + 2), self.metrics['val_' + metric], label='val_' + metric,
                                 color='orange', )
                if metric == 'accuracy':
                    self.axs[i].set_title(
                        f"{'Max accuracy': <25}: {self.max_acc:.4f}, epoch {self.acc_ep}\n{'Max val_accuracy': <25}: {self.max_val_acc:.4f}, epoch {self.val_acc_ep}")
                elif metric == 'loss':
                    self.axs[i].set_title(
                        f"{'Min loss': <25}: {self.min_loss:.4f}, epoch {self.loss_ep}\n{'Min val_loss': <25}: {self.min_val_loss:.4f}, epoch {self.val_loss_ep}")
            if self.first_epoch:
                self.axs[i].legend()
                self.axs[i].grid()
        self.first_epoch = False
        plt.tight_layout()
        self.f.canvas.draw()
        self.f.canvas.flush_events()


def main():
    labels = """airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck""".split("\n")
    animal_labels = np.array([2, 3, 4, 5, 6])
    vehicle_labels = np.array([0, 1, 7, 8, 9])
    (x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()
    labels = vehicle_labels

    y_train_all, y_test_all = y_train_all.flatten(), y_test_all.flatten()
    x_train = x_train_all[np.isin(y_train_all, labels, ).ravel()]
    y_train = y_train_all[np.isin(y_train_all, labels, ).ravel()]
    x_test = x_test_all[np.isin(y_test_all, labels, ).ravel()]
    y_test = y_test_all[np.isin(y_test_all, labels, ).ravel()]

    # x_train = np.expand_dims(x_train, axis=-1)
    # x_test = np.expand_dims(x_test, axis=-1)

    classes = np.unique(y_train)
    print(classes)
    for i, v in enumerate(labels):
        y_train[y_train == v] = i
    for i, v in enumerate(labels):
        y_test[y_test == v] = i
    y_test = to_categorical(y_test, len(classes))
    y_train = to_categorical(y_train, len(classes))

    K = len(classes)

    print("number of classes:", K)
    print("input shape", x_train[0].shape)
    model = cifar_model(x_train[0].shape, K)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, shuffle=True)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.05,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    )  # randomly flip images
    validgen = ImageDataGenerator(rescale=1. / 255, )

    train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)
    valid_gen = validgen.flow(x_val, y_val, batch_size=batch_size)
    history = model.fit(
        train_gen,
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=x_val.shape[0] // batch_size,
        callbacks=[
            EarlyStopping(monitor="val_loss", verbose=1, patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=6, verbose=1, min_lr=0.00001),
            PlotProgress("data"),
        ],
    )
    model.evaluate(x_train/255, y_train)
    model.evaluate(x_test/255, y_test)
    model.save("cifar-vehicles")

    plt.plot(history.history['accuracy'], label='acc', color='red')
    plt.plot(history.history['val_accuracy'], label='val_acc', color='green')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
