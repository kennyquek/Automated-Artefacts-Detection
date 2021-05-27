import tensorflow as tf
from tensorflow.keras import applications, Model, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import SGD, Adam

IMG_HEIGHT = 100
IMG_WIDTH = 100
batch_size = 5
epochs = 30

PATH = "C:/Users/kenny/Desktop/dataset structural"

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

base_model = applications.vgg16.VGG16(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
# 0.2
x = layers.Dropout(0.5)(x)
# predictions = layers.Dense(number_of_class, activation= 'softmax')(x)
x = layers.Flatten()(x)  # Becomes a vector
x = layers.Dense(512, activation='relu')(x)  # There is no specific manner of selecting the number of units in a dense layer. You need to try different units and see which gives the best accuracy and the lowest loss value
# x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
predictions = layers.Dense(1, activation='sigmoid')(x)  # sigmoid is used for the two-class logistic regression

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# change adam to sgd to increase learning rate due to over fitting SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True),
model.compile(optimizer= Adam(learning_rate= 0.000001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_data_gen,
          epochs=epochs, steps_per_epoch=300/batch_size,
          validation_data=val_data_gen,
          validation_steps= 60/batch_size)

# save whole model
model.save('C:/Users/kenny/Desktop/checkpoint/whole model/pretrained_VGG16_model_stuctural.h5')
