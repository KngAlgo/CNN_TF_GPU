import tensorflow as tf
tf.config.experimental.enable_mlir_graph_optimization = False

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow import keras
from keras import layers, Model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_dataset = keras.utils.image_dataset_from_directory('/workspace/CNN Classifier/images/train', batch_size=16)

test_dataset = keras.utils.image_dataset_from_directory('/workspace/CNN Classifier/images/test', batch_size=16)

val_dataset = keras.utils.image_dataset_from_directory('/workspace/CNN Classifier/images/valid', batch_size=16)

data_aug = keras.Sequential(
    [
        layers.RandomFlip('horizontal'), 
        layers.RandomZoom(0.2), 
        layers.RandomRotation(0.1)
        ]
    )

def make_model():
    input = keras.Input(shape=(256, 256, 3))
    x = data_aug(input)
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    
    output = layers.Dense(53, activation='softmax')(x)
    
    model = keras.Model(input, output)

    return model

model_ = make_model()

checkpoint = keras.callbacks.ModelCheckpoint("card_model.keras", monitor='val_loss')

model_.compile(optimizer='adam', metrics=['accuracy'], loss="sparse_categorical_crossentropy")

model_.fit(train_dataset, epochs=12, validation_data=val_dataset, callbacks=[checkpoint])

# # model = keras.load_model('card_model.keras')

# pred = model.predict(test_dataset)