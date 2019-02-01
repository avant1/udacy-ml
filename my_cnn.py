#%%

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# session = tf.Session(config=config)


#%%

import os

from keras.models import Sequential

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


#%%

model = Sequential()

image_dimentions = 64

model.add(Conv2D(filters=32, kernel_size=3, padding='same', input_shape=(image_dimentions, image_dimentions, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())


#%%

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%%

current_dir = os.path.dirname(__file__)
base_dir = current_dir + '/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/'

#%%

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
training_set = train_datagen.flow_from_directory(
    base_dir + 'training_set',
    target_size=(image_dimentions, image_dimentions),
    batch_size=batch_size,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    base_dir + 'test_set',
    target_size=(image_dimentions, image_dimentions),
    batch_size=batch_size,
    class_mode='binary')

#%%
model.fit_generator(
    training_set,
    steps_per_epoch=int(training_set.samples / batch_size),
    epochs=10,
    validation_data=test_set,
    validation_steps=test_set.samples)


#%%
