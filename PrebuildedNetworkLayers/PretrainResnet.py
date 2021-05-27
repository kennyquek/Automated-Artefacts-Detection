import tensorflow as tf
from tensorflow.keras import applications, Model, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam

IMG_HEIGHT = 100
IMG_WIDTH = 100
batch_size = 10
epochs = 30

PATH = "C:/Users/kenny/Desktop/Whole angio"

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_defects_dir = os.path.join(train_dir, 'defects')  # directory with our training cat pictures
train_normal_dir = os.path.join(train_dir, 'normal')

validation_defects_dir = os.path.join(validation_dir, 'defects')
validation_normal_dir = os.path.join(validation_dir, 'normal')

train_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

# Did some data argumentation as dataset is abit little

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

x = base_model.output
x = layers.MaxPooling2D()(x)
# Prevent over fitting
x = layers.Dropout(0.2)(x)
# Might need to add another dense layer (try 4026)
x = layers.Flatten()(x)  # Becomes a vector
x = layers.Dense(512, activation='relu')(x)  # There is no specific manner of selecting the number of units in a dense layer. You need to try different units and see which gives the best accuracy and the lowest loss value
# To be able to increase the learning rate so as to save more time
x = layers.BatchNormalization()(x)
predictions = layers.Dense(1, activation='sigmoid')(x)  # sigmoid is used for the two-class logistic regression

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# change adam to sgd to increase learning rate due to over fitting SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True),
model.compile(optimizer= Adam(lr=0.0000001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_data_gen,
          epochs=epochs, steps_per_epoch=700/batch_size,
          validation_data=val_data_gen,
          validation_steps= 40/batch_size)

# save whole model
# model.save('C:/Users/kenny/Desktop/checkpoint/whole model/pretrained_resnet_50_2_model.h5')

main_path = "C:/Users/kenny/Desktop/Whole angio/validation/defects/"
files_name_array = os.listdir(main_path)
# predict model with validation set
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

print("Defect model scores " + str(points) + " /20")

main_path = "C:/Users/kenny/Desktop/Whole angio/validation/normal/"

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

print("Normal model scores " + str(points) + " /20")