# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from datetime import timedelta, datetime

model_name = 'c32c32c64fc64'

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', name='Conv1'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu', name='Conv2'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu', name='Conv3'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu', name='FC1'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Loading the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size = 32
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = batch_size,
                                            class_mode = 'binary')

# Callbacks during training
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from time import time
import os

class MyCallback(Callback):
    def on_train_begin(self, logs={}):
        print('MyCallback : let\'s train')

    def on_epoch_end(self, epoch, logs={}):
        print('\nMyCallback : End of epoch')

checkpointer = ModelCheckpoint('models/model_epoch-{epoch:02d}-{val_acc:.2f}.hd5f', period=1, save_best_only=True)
logdir = 'logdir/{}'.format(time())
if not os.path.exists(logdir):
    os.makedirs(logdir)
# tb = TensorBoard(log_dir=logdir, write_graph=False, histogram_freq=1, batch_size=32, write_images=False )
mc = MyCallback()


#eventually load weights
# classifier.load_weights('models/final_weights.h5', by_name=True)

# callbacks = [checkpointer, tb(validation_data=test_set), mc(test_set)]
callbacks = [checkpointer, mc]
t0 = datetime.today()
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000 / batch_size,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000 / batch_size,
                         verbose = 2,
                         callbacks=callbacks)

td = datetime.today() - t0
with open('models/{}.json'.format(model_name), 'w') as f:
    f.write(classifier.to_json())
classifier.save_weights('models/{}final_weights.h5'.format(model_name))


print('Training took {}'.format(td))

t0 = datetime.today()
score = classifier.evaluate_generator(test_set)
td = datetime.today() - t0
print('evaluation took', td, 'for', test_set.samples, 'samples')
for idx, metric in enumerate(classifier.metrics_names):
    print(metric, ' : ', score[idx])