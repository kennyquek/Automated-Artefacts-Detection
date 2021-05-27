import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os

checkpoint_path = "C:/Users/kenny/Desktop/checkpoint/whole model/pretrained_VGG16_model.h5"
test_dir = "C:/Users/kenny/Desktop/dataset structural/test"
IMG_HEIGHT = 100
IMG_WIDTH = 100

# model = Sequential([
#     Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
#     MaxPooling2D(),
#     Conv2D(32, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
#
# model.load_weights(checkpoint_path)
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

model = tf.keras.models.load_model(checkpoint_path)

# test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
#
# test_data_gen = test_image_generator.flow_from_directory(
#                                                            directory=test_dir,
#                                                            shuffle=True,
#                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                            class_mode='binary')
#
# loss,acc = model.evaluate(test_data_gen, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# {'defects': 0, 'normal': 1}
# Predicting normal images

main_path = "C:/Users/kenny/Desktop/dataset structural/test/defects/"

files_name_array = os.listdir(main_path)

# print(str(files_name_array))
# defects file index is 0
points = 0
for defects_file_name in files_name_array:
    test_image = tf.keras.preprocessing.image.load_img(
        main_path + defects_file_name
        , target_size=(IMG_HEIGHT, IMG_WIDTH))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    class_index = int(result[0][0])
    print(defects_file_name + " = " + str(class_index))
    if(class_index == 0):
        points += 1

print("Defect model scores " + str(points) + " /10")

main_path = "C:/Users/kenny/Desktop/dataset structural/test/normal/"

files_name_array = os.listdir(main_path)

# print(str(files_name_array))
# normal file index is 1
points = 0
for normal_file_name in files_name_array:
    test_image = tf.keras.preprocessing.image.load_img(
        main_path + normal_file_name
        , target_size=(IMG_HEIGHT, IMG_WIDTH))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    class_index = int(result[0][0])
    print(normal_file_name + " = " + str(class_index))
    if (class_index == 1):
        points += 1

print("Normal model scores " + str(points) + " /10")