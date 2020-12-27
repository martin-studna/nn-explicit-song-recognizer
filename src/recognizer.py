import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
import argparse

"""



"""

batch_size = 100
epochs = 100
 
# Load data
data = pd.read_csv("../data/data.csv")

# === PREPROCESSING ===
 
# Remove unnecessary data
data = data.drop(['name'], axis=1).drop(['release_date'], axis=1).drop(['id'], axis=1).drop(['artists'], axis=1)

# Normalize
min_max_scaler = preprocessing.MinMaxScaler()
data = pd.DataFrame(min_max_scaler.fit_transform(data.values))
 
# Show column names of the table.
print(data.head())
 

# All explicit data contains True in Explicit column. 
explicit_data = data.values[np.where(data.values[:, 6] == 1)]
# Non-explicit data contains False in Explicit column.
non_explicit_data = data.values[np.where(data.values[:, 6] == 0)]
# Select same number of non-explicit data as explicit data. 
non_explicit_data = non_explicit_data[0:len(explicit_data)]


# Devide explicit data on Training set and Test set. 
explicit_train, explicit_test = train_test_split(explicit_data, test_size=0.2, random_state=1)
# Devide non-explicit data on Training set and Test set.
non_explicit_train, non_explicit_test = train_test_split(non_explicit_data, test_size=0.2, random_state=1)
 
# Concatenate training data and test data.
train = np.concatenate((explicit_train, non_explicit_train))
test = np.concatenate((explicit_test, non_explicit_test))

np.random.shuffle(train)
np.random.shuffle(test)
 
train = pd.DataFrame(train)
test = pd.DataFrame(test)
 
train_X = train.drop(6, axis=1).values
train_Y = train[6].values
 
test_X = test.drop([6], axis=1).values
test_Y = test[6].values
 
#train_X = np.delete(train, 6, axis=1)
#train_Y = values[:, 6]
 
# create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(14, activation=tf.nn.sigmoid, input_shape=(14,)),
    tf.keras.layers.Dense(100, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
 
model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
 
# train
tb_callback = tf.keras.callbacks.TensorBoard('true', update_freq=1000, profile_batch=0)
model.fit(
    train_X, train_Y,
    batch_size=batch_size, epochs=epochs,
    callbacks=[tb_callback]
)
 
result = model.predict(test_X)
result_Y = (result > 0.5).flatten()
 
conf_matrix = tf.math.confusion_matrix((test_Y > 0.5).flatten(), result_Y)

sns.set(font_scale=1.4) # for label size
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}) # font size

plt.show()
print(result_Y)