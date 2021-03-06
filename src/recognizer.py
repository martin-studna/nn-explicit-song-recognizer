import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
import argparse
import confusion_matrix_pretty_print

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
parser.add_argument("--plot_conf", default=True, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")

"""
    Plot confusion matrix method
"""


def plot_conf_matrix(predictions, outputs):
    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(
        outputs, predictions)


def train_network(train_X, train_Y, test_X, test_Y, hidden_size, batch_size, epochs, lr, plot_conf):
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            train_X.shape[1], activation=tf.nn.sigmoid, input_shape=(train_X.shape[1],)),
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

    if (plot_conf):
        plot_conf_matrix(predictions, outputs)


def main(args):

    # Load data Spotify dataset
    data = pd.read_csv("../data/data.csv")

    # Remove unnecessary data
    data = data.drop(['name', 'release_date', 'id',
                      'artists'], axis=1)

    explicit_column_index = data.columns.get_loc("explicit")

    # Normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    data = pd.DataFrame(min_max_scaler.fit_transform(data.values))

    '''
       First task - same number of positive and negative examples
    '''
    # All explicit data contains True in Explicit column.
    explicit_data = data.values[np.where(
        data.values[:, explicit_column_index] == 1)]
    # Non-explicit data contains False in Explicit column.
    non_explicit_data = data.values[np.where(
        data.values[:, explicit_column_index] == 0)]
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

    train_X = train.drop(explicit_column_index, axis=1).values
    train_Y = train[explicit_column_index].values

    test_X = test.drop([explicit_column_index], axis=1).values
    test_Y = test[explicit_column_index].values

    train_network(train_X, train_Y, test_X, test_Y,
        args.hidden_size, args.batch_size, args.epochs, args.lr, args.plot_conf)

    '''
        Second task - with all data - 8.5% negative examples
    '''
    train, test = train_test_split(data, test_size=args.test_size, random_state=args.seed)

    train_X = train.drop(explicit_column_index, axis=1).values
    train_Y = train[explicit_column_index].values

    test_X = test.drop([explicit_column_index], axis=1).values
    test_Y = test[explicit_column_index].values

    train_network(train_X, train_Y, test_X, test_Y,
        args.hidden_size, args.batch_size, args.epochs, args.lr, args.plot_conf)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
