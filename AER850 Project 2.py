# %% step 1

# import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define image shape
IMG_WIDTH, IMG_HEIGHT, CHANNELS = 100, 100, 3

# establish the train and validation data directories
train_data_dir = './Train'
validation_data_dir = './Validation'

# define data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# create the train and validation data generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=32,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=32,
    class_mode='categorical')

# %% Step 2

"""

###

Here's a simple implementation of such a neural network in TensorFlow/Keras. This model consists of a few Conv2D layers followed by MaxPooling2D, 
flattening and fully connected layers. The architecture is such that there are several Conv2D layers with MaxPooling2D layers after them. There 
is a Flatten layer which then leads onto fully connected Dense layers, ending with a Dense output layer with 4 neurons (corresponding to 4 classes). 
Activations used are rectified linear units (ReLUs) for hidden layers and Softmax for the output layer. Dropout layer is also used to avoid overfitting.

###

# import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

# define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # flattening the 2D arrays for fully connected layers
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4)) # The output layer with 4 neurons, for 4 classes
model.add(Activation('softmax'))

"""

# %% Step 3

#Here is my inital model without  specificying any hyperparamters. (Not fine tuned for the project and therfore less accurate)
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.layers import Flatten, Dense

# define the model architecture
#model = Sequential()

#model.add(Conv2D(input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)))
#model.add(MaxPooling2D())

#model.add(Conv2D())
#model.add(MaxPooling2D())

#model.add(Conv2D())
#model.add(MaxPooling2D())

#model.add(Flatten())  # Flatten layer
#model.add(Dense())

#model.add(Dense()) # Output layer


# This is the model I will be using for this project:
    
# import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

# define the model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # Flatten layer
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(4)) # Output layer
model.add(Activation('softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#In this code, ‘LeakyReLU’ activation function has been included in the Conv2D, dense layers for adding non-linearity and better generalization in the neural network. The output layer uses the ‘softmax’ activation function for a multi-class classification problem. The number of filters in the Conv2D layers are incremented in powers of 2, as are the neurons in the dense layers.
# Finally, the model is compiled using the 'categorical crossentropy' loss function, which is suitable for multi-class classification, and 'Adam' as the optimizer. Adam is a very effective and widely used optimizer. The metric we aim to optimize is 'accuracy'.

# %% Step 4

# fit the model and get the training data statistics 

import matplotlib.pyplot as plt

history = model.fit(
        train_generator,
        epochs=50,
        steps_per_epoch=50,  # set to total training data/batch size
        validation_data=validation_generator,
        validation_steps=25)  # set to total validation data/batch size


# summarize the history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# summarize the history of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# %% Step 5

# save model to HDF5
model.save('CNN_model.h5')








