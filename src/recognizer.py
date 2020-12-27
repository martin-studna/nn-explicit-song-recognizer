import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
import argparse

"""Explicit Song Recognizer Project
    
    Martin Studna
    Jan Babušík


"""


parser = argparse.ArgumentParser()

parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
parser.add_argument("--lr", default=0.05, type=int, help="Learning rate")
parser.add_argument("--hidden_size", default=100, type=int,
                    help="Size of the hidden layer")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--batch_size", default=100, type=int, help="Batch size")
parser.add_argument("--test_size", default=0.2, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")

"""
    Confusion matrix statistics
"""


def conf_matrix_stats(result, test):
    a = 0
    b = 0
    c = 0
    d = 0

    for i in range(len(test)):
        if (result[i] == False and test[i] == 0):
            a += 1
        if (result[i] == False and test[i] == 1):
            b += 1
        if (result[i] == True and test[i] == 0):
            c += 1
        if (result[i] == True and test[i] == 1):
            d += 1

    print(a)
    print(b)
    print(c)
    print(d)


"""
    Plot confusion matrix method
"""


def plot_conf_matrix(predictions, outputs):
    conf_matrix = tf.math.confusion_matrix(predictions, outputs)

    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16})  # font size

    plt.show()


def train_network(train_X, train_Y, test_X, test_Y, hidden_size, batch_size, epochs, lr):
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            train_X.shape[1], activation=tf.nn.sigmoid, input_shape=(14,)),
        tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # train
    tb_callback = tf.keras.callbacks.TensorBoard(
        'true', update_freq=1000, profile_batch=0)
    model.fit(
        train_X, train_Y,
        batch_size=batch_size, epochs=epochs,
        callbacks=[tb_callback]
    )

    results = model.predict(test_X)
    predictions = (results > 0.5).flatten()
    outputs = (test_Y > 0.5).flatten()

    conf_matrix_stats(predictions, outputs)
    plot_conf_matrix(predictions, outputs)


def main(args):

    # Load data Spotify dataset
    data = pd.read_csv("../data/data.csv")

    # === PREPROCESSING ===

    # Remove unnecessary data
    data = data.drop(['name', 'release_date', 'id',
                      'artists'], axis=1)

    # Normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    data = pd.DataFrame(min_max_scaler.fit_transform(data.values))

    # All explicit data contains True in Explicit column.
    explicit_data = data.values[np.where(data.values[:, 6] == 1)]
    # Non-explicit data contains False in Explicit column.
    non_explicit_data = data.values[np.where(data.values[:, 6] == 0)]
    # Select same number of non-explicit data as explicit data.
    non_explicit_data = non_explicit_data[0:len(explicit_data)]

    # Devide explicit data on Training set and Test set.
    explicit_train, explicit_test = train_test_split(
        explicit_data, test_size=args.test_size, random_state=args.seed)
    # Devide non-explicit data on Training set and Test set.
    non_explicit_train, non_explicit_test = train_test_split(
        non_explicit_data, test_size=args.test_size, random_state=args.seed)

    # Concatenate training data and test data.
    train = np.concatenate((explicit_train, non_explicit_train))
    test = np.concatenate((explicit_test, non_explicit_test))

    # Shuffle rows (records)
    np.random.shuffle(train)
    np.random.shuffle(test)

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    train_X = train.drop(6, axis=1).values
    train_Y = train[6].values

    test_X = test.drop([6], axis=1).values
    test_Y = test[6].values

    train_network(train_X, train_Y, test_X, test_Y,
                  args.hidden_size, args.batch_size, args.epochs, args.lr)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
