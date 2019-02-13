import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    y_train = train["label"]
    train = train.drop(labels=["label"], axis=1)

    train = train / 255.0
    test = test / 255.0

    train = train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    y_train = to_categorical(y_train, num_classes=10)

    X_train, X_val, y_train, y_val = train_test_split(train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    model = load_model()

    train_model(model, X_train, X_val, y_train, y_val)

    save_model(model)

    results = model.predict(test)
    results = np.argmax(results, axis=1)

    results = pd.Series(results, name="Label")

    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

    submission.to_csv("submission.csv", index=False)


def train_model(model, X_train, X_val, y_train, y_val):
    optimizer = RMSprop()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    learning_rate_reducer = ReduceLROnPlateau(monitor="val_acc", patience=3, verbose=1, factor=0.3, min_lr=0.00001)
    epochs = 1
    batch_size = 86
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                        verbose=2)
    return model


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


def get_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    return model


if __name__ == "__main__":
    main()
