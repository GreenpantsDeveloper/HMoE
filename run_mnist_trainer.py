import os, sys
import numpy as np

from tensorflow.keras.layers import Input
from keras.datasets import mnist
from keras.utils import to_categorical
from math import floor

#from hmoe import build_HMoE, build_MLP
from hmoe import build_HMoE
from config import *
from utils import get_cb_checkpoint, get_cb_early_stopping, get_cb_cyclic_lr


if __name__ == '__main__':

    # Train on MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Scale data to [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Reshape data to 'flatten' images
    x_train = np.reshape(x_train, (x_train.shape[0], INPUT_SIZE))
    x_test = np.reshape(x_test, (x_test.shape[0], INPUT_SIZE))

    # Make labels categorical
    y_train = to_categorical(y_train, OUTPUT_SIZE)
    y_test = to_categorical(y_test, OUTPUT_SIZE)

    print("\nTrain:")
    print(x_train.shape)
    print(y_train.shape)
    print("\nTest:")
    print(x_test.shape)
    print(y_test.shape)

    ## Build & compile the Hierarchical Mixture of Experts with parameters given in `config.py`
    model = build_HMoE()
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    model.summary()


    #######################################


    ### Train the HMoE model
    print("\nModel initialized; let's start training!")

    # Get callbacks
    callbacks = []
    callbacks.append(get_cb_early_stopping(patience=EARLY_STOPPING_PATIENCE))
    callbacks.append(get_cb_checkpoint(model=model, save_file=(MODEL_PATH + MODEL_NAME + '_model.hdf5')))
    callbacks.append(get_cb_cyclic_lr())

    # Train the model (see https://keras.io/api/models/model_training_apis/)
    hist = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        validation_data=(x_test, y_test)
    )

    print("\n### Training is done! ###\n")

    # Store the history data
    hist_filepath = MODEL_PATH + MODEL_NAME + '_data.npy'
    print("Saving history to `%s`..." % hist_filepath)
    val_acc = hist.history['val_accuracy']
    np.save(hist_filepath, np.array(val_acc))
    print("History saved!")
