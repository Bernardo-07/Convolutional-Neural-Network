import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import imghdr

def normalizer(image, label): 
    image = tf.cast(image, dtype=tf.float32)/255.0
    label = tf.one_hot(label, depth=2)
    return image, label

dir = "dataset/training"

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for class_name in os.listdir(dir):  
    class_path = os.path.join(dir, class_name)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        if imghdr.what(img_path) not in ['jpeg', 'png', 'jpg']:  # Verifica se é uma imagem
            print(f"Arquivo não suportado ou corrompido: {img_name}")
            continue
    
        try:
            img = load_img(img_path)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            count = 0
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=class_path, save_prefix=f"aug_{img_name.split('.')[0]}", save_format="jpg"):
                count += 1
                if count >= 2: 
                    break
        except Exception as e:
            print(f"Erro ao carregar {img_name}: {e}")
            continue

train_data, valid_data = tf.keras.utils.image_dataset_from_directory(
    'dataset/training',
    validation_split=0.15,  
    subset="both",     
    seed=42,               
    image_size=(128, 128),
    shuffle=True, 
    batch_size=32          
)

test_data = tf.keras.utils.image_dataset_from_directory(
    'dataset/test',                 
    image_size=(128, 128), 
    batch_size=32          
)

test = test_data.map(normalizer)

train = train_data.map(normalizer)
valid = valid_data.map(normalizer) 

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=2, activation='softmax'))


model.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
)

print(model.summary())

learning_rate_reduction = ReduceLROnPlateau(
    monitor = 'val_accuracy',
    patience=2,
    factor=0.5,
    min_lr = 0.00001,
    verbose = 1
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose= 0)

hist = model.fit(
    train,
    validation_data=valid, 
    epochs=30,  
    batch_size=32,
    shuffle=True,
    callbacks=[early_stopping,learning_rate_reduction],
    verbose = 1
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

from tensorflow.keras.models import load_model

model.save(os.path.join('models', 'softmax.h5'))

new_model = load_model(os.path.join('models', 'softmax.h5'))

loss, acc = new_model.evaluate(test, batch_size=32)

print(loss)
print(acc)

y_true = []
for images, labels in test:
    y_true.extend(labels.numpy())  # Convertendo labels para array NumPy

y_true = np.array(y_true)

result = new_model.predict(test)

y_pred = (result > 0.5).astype(int)

import seaborn as sns

matrix = tf.math.confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(
    matrix.numpy(), 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=['Cat', 'Dog'], 
    yticklabels=['Cat', 'Dog'],
    vmin = 400
)

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()