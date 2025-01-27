import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def normalizer(image, label):
    aux = tf.cast(image, dtype=tf.float32)
    image_norm = aux/255.0
    return image_norm, label

train_data = tf.keras.utils.image_dataset_from_directory(
    'dataset/training',
    validation_split=0.1,  
    subset="training",     
    seed=42,               
    image_size=(256, 256), 
    batch_size=32          
)

valid_data = tf.keras.utils.image_dataset_from_directory(
    'dataset/training',
    validation_split=0.1,  
    subset="validation",   
    seed=42,               
    image_size=(256, 256), 
    batch_size=32          
)

train = train_data.map(normalizer)
valid = valid_data.map(normalizer) 

model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(MaxPooling2D())

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'],
)

print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

hist = model.fit(
    train,
    batch_size=32, 
    epochs=20, 
    validation_data=valid,
    callbacks=[early_stopping], 
    verbose=1
)

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='valid')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss in Training and Validation')
plt.show()

if 'accuracy' in hist.history:  
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='valid')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy in Training and Validation')
    plt.show()