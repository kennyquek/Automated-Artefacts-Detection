import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

import os

def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  # the output is equal to input BUT MUST BE SAME SHAPE
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

IMG_HEIGHT = 100
IMG_WIDTH = 100
batch_size = 5 # Since dataset is 150, train 30 times
epochs = 12

number_of_class = 2


PATH = "C:/Users/kenny/Desktop/dataset angio"

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_defects_dir = os.path.join(train_dir, 'defects')  # directory with our training cat pictures
train_normal_dir = os.path.join(train_dir, 'normal')

validation_defects_dir = os.path.join(validation_dir, 'defects')
validation_normal_dir = os.path.join(validation_dir, 'normal')

num_defects_tr = len(os.listdir(train_defects_dir))
num_normal_tr = len(os.listdir(train_normal_dir))

num_defects_val = len(os.listdir(validation_defects_dir))
num_normal_val = len(os.listdir(validation_normal_dir))


total_train = num_defects_tr
total_val = num_defects_val

print('total training defects images:', num_defects_tr)
print('total training normal images:', num_normal_tr)

print('total validation defects images:', num_defects_val)
print('total validation normal images:', num_normal_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

# Data argumentation to increase accuracy!

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)

num_res_net_blocks = 10

for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)

x = layers.Conv2D(64, 3, activation='relu')(x)
# Global Average Pooling is an operation that calculates the average output of each feature map in the previous layer.
# This fairly simple operation reduces the data significantly and prepares the model for the final classification layer
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(number_of_class, activation='softmax')(x)

res_net_model = tf.keras.Model(inputs, outputs)

# binary classification (two target classes)
# multi-class classification (more than two exclusive targets)
# multi-label classification (more than two non exclusive targets) in which multiple target classes can be on at the same time

# optimizer https://keras.io/optimizers/ adam is the default on

res_net_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# res_net_model.summary()

history = res_net_model.fit_generator(train_data_gen, epochs=epochs, steps_per_epoch=batch_size,
          validation_data=val_data_gen,
          validation_steps=batch_size)

# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# save whole model
res_net_model.save('C:/Users/kenny/Desktop/checkpoint/whole model/resnet_model.h5')