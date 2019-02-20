import binascii
from collections import deque

import numpy as np
import sys
from sklearn.utils import class_weight
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D
from keras.regularizers import l1_l2
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt

def load_traces(database_file, start_at=0, number_samples=0):
    traces = np.loadtxt(database_file, delimiter=',', dtype=np.float64, skiprows=1,
                        usecols=range(start_at, number_samples))
    inputoutput = np.loadtxt(database_file, delimiter=',', dtype=np.str, skiprows=1,
                             usecols=start_at - 1)
    # print("traces shape: {}\ninputoutput shape: {}\n".format(traces.shape, inputoutput.shape))
    return traces, inputoutput


def shorten_traces(dataset, start_at=0, number_samples=15000):
    if len(dataset) == 2:
        traces, inputoutput = dataset
    elif len(dataset) == 3:
        traces, inputoutput, labels = dataset

    traces_selected = traces[start_at:start_at + number_samples]

    if len(dataset) == 2:
        return traces_selected, inputoutput
    elif len(dataset) == 3:
        return traces_selected, inputoutput, labels


def statcorrect_traces(dataset):
    if len(dataset) == 2:
        traces, inputoutput = dataset
    elif len(dataset) == 3:
        traces, inputoutput, labels = dataset

    # traces_statcorrect = (traces - np.mean(traces, axis=1).reshape(-1,1))/np.std(traces, axis=1).reshape(-1,1)
    traces_statcorrect = (traces - np.mean(traces, axis=0).reshape(1, -1)) / np.std(traces, axis=0).reshape(1, -1)

    if len(dataset) == 2:
        return traces_statcorrect, inputoutput
    elif len(dataset) == 3:
        return traces_statcorrect, inputoutput, labels


# noinspection PyShadowingNames
def split_data_percentage(dataset, training_fraction=0.5):
    if len(dataset) == 2:
        traces, inputoutput = dataset
    elif len(dataset) == 3:
        traces, inputoutput, labels = dataset

    traces_train = traces[:int(traces.shape[0] * training_fraction)]
    traces_test = traces[int(traces.shape[0] * training_fraction):]
    inputoutput_train = inputoutput[:int(inputoutput.shape[0] * training_fraction)]
    inputoutput_test = inputoutput[int(inputoutput.shape[0] * training_fraction):]
    if len(dataset) == 3:
        labels_train = labels[:int(labels.shape[0] * training_fraction)]
        labels_test = labels[int(labels.shape[0] * training_fraction):]

    if len(dataset) == 2:
        return (traces_train, traces_test), (inputoutput_train, inputoutput_test)
    elif len(dataset) == 3:
        return (traces_train, traces_test), (inputoutput_train, inputoutput_test), (labels_train, labels_test)


def create_labels_rOut(dataset, database_file, col, static_key=0):
    if len(dataset) == 2:
        traces, inputoutput = dataset
    elif len(dataset) == 3:
        traces, inputoutput, a = dataset

    labels = np.loadtxt(database_file, delimiter=',', dtype=np.int32, skiprows=1, usecols=col+static_key)
    return traces, inputoutput, labels


# use for hamming weight leakage model
def create_model(classes=9, number_samples=200):
    input_shape = (number_samples, 1)
    trace_input = Input(shape=input_shape)
    x = Conv1D(filters=10, kernel_size=10, strides=10, activation='relu', padding='valid', name='block1_conv1')(
        trace_input)
    x = MaxPooling1D(pool_size=1, strides=1, padding='valid', name='block1_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(50, activation='tanh', name='fc1')(x)
    x = Dense(50, activation='tanh', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='cnn')
    optimizer = SGD(lr=0.01, decay=0, momentum=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# use for bit leakage model
def create_big_model(classes=256, number_samples=200):
    input_shape = (number_samples, 1)
    trace_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(filters=64, kernel_size=10, strides=10, activation='relu', padding='same', name='block1_conv1')(
        trace_input)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same', name='block1_pool')(x)
    # Block 2
    x = Conv1D(filters=128, kernel_size=10, strides=10, activation='relu', padding='same', name='block2_conv1')(
        x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same', name='block2_pool')(x)
    # Block 3
    x = Conv1D(filters=256, kernel_size=10, strides=10, activation='relu', padding='same', name='block3_conv1')(
        x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same', name='block3_pool')(x)
    # Block 4
    x = Conv1D(filters=512, kernel_size=10, strides=10, activation='relu', padding='same', name='block4_conv1')(
        x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same', name='block4_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(50, activation='tanh', name='fc1')(x)
    x = Dense(50, activation='tanh', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='cnn')
    optimizer = SGD(lr=0.01, decay=0, momentum=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':

    if 'dataset' not in locals():
        dataset = load_traces('traceHWDES.csv', 1, 70)

        dataset = statcorrect_traces(dataset)
        # dataset = shorten_traces(dataset, 228405-100, 200)

        # Args: Dataset, Byte to Attack, Subkey[Bayte] of first round
        test_acc = np.zeros(8)
        for i in range(8):
            datasetLeak = create_labels_rOut(dataset, 'traceHWDES.csv', 72, i)  # Template - Use calculated inter values
            datasetLeak = split_data_percentage(datasetLeak, training_fraction=0.85)
            (traces_train, traces_test), (inputoutput_train, inputoutput_test), (labels_train, labels_test) = datasetLeak

            print("<------------------Using byte: {}------------------>".format(i))
            # print(traces_train.shape, traces_train.dtype)
            # print(traces_test.shape, traces_test.dtype)
            # print(inputoutput_train.shape, inputoutput_train.dtype)
            # print(inputoutput_test.shape, inputoutput_test.dtype)
            # print(labels_train.shape, labels_train.dtype)
            # print(labels_test.shape, labels_test.dtype)
            # print(labels_train[0])

            min_class_tr = int(np.min(labels_train))
            min_class_ts = int(np.min(labels_test))
            classes = max(len(np.unique(labels_train)) + min_class_tr, len(np.unique(labels_test)) + min_class_ts)
            classes = 9

            traces_train_reshaped = traces_train.reshape((traces_train.shape[0], traces_train.shape[1], 1))
            labels_train_categorical = to_categorical(labels_train, num_classes=classes)
            traces_test_reshaped = traces_test.reshape((traces_test.shape[0], traces_test.shape[1], 1))
            labels_test_categorical = to_categorical(labels_test, num_classes=classes)

            save_model = ModelCheckpoint('model_epoch{epoch}.h5', period=100)


            class CalculateRecall(Callback):
                def __init__(self, data, labels, message_prefix=None):
                    self.data = data
                    self.labels = labels
                    self.message_prefix = message_prefix + ' ' or ''

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}

                    predictions = self.model.predict(self.data)
                    correctly_classified = (np.argmax(predictions, axis=1) == self.labels)
                    _sum = 0.
                    for i in np.unique(self.labels):
                        n_correct = len(np.nonzero(correctly_classified[np.where(self.labels == i)[0]])[0])
                        n_total = len(np.where(self.labels == i)[0])
                        _sum += n_correct / n_total
                    recall = _sum / len(np.unique(self.labels))

                    print(self.message_prefix + 'recall:', recall)


            calculate_recall_train = CalculateRecall(traces_train_reshaped, labels_train, 'train')
            calculate_recall_test = CalculateRecall(traces_test_reshaped, labels_test, 'test')
            callbacks = [calculate_recall_train, calculate_recall_test, save_model]

            model = create_model(classes=classes, number_samples=traces_train.shape[1])

            history = model.fit(x=traces_train_reshaped,
                                y=labels_train_categorical,
                                batch_size=3000,
                                verbose=0,
                                epochs=800,
                                class_weight=class_weight.compute_class_weight('balanced', np.unique(labels_train),
                                                                               labels_train),
                                validation_data=(traces_test_reshaped, labels_test_categorical),
                                callbacks=callbacks)

            t = model.evaluate(x=traces_test_reshaped,
                               y=labels_test_categorical,
                               verbose=0)

            test_acc[i] = t[1]

        print(test_acc)
