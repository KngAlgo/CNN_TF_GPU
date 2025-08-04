import tensorflow as tf

from tensorflow import keras
from keras import layers, Model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

input = keras.Input(shape=(256, 256, 3))
x = layers.Rescaling(1./255)(input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Flatten()(x)

output = layers.Dense(53, activation='softmax')(x)

model = keras.Model(input, output)

train_dataset = keras.utils.image_dataset_from_directory('/workspace/CNN Classifier/images/train', batch_size=32)

test_dataset = keras.utils.image_dataset_from_directory('/workspace/CNN Classifier/images/test', batch_size=32)

val_dataset = keras.utils.image_dataset_from_directory('/workspace/CNN Classifier/images/valid', batch_size=32)

checkpoint = keras.callbacks.ModelCheckpoint("/card_model.keras", save_best_only=True)

model.compile(optimizer='adam', metrics=['accuracy'], loss="sparse_categorical_crossentropy")

model.fit(train_dataset, epochs=25, validation_data=val_dataset, callbacks=[checkpoint])