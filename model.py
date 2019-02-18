import csv
import  cv2
import numpy as np
import sklearn

np.random.seed(0)

def read_csv(path):
    lines = []
    with open(path + "/driving_log.csv", "r") as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines

def read_images(data_path, lines):
    images = []
    measurements = []
    offset = [0, 0.2, -0.2]
    for i in range(1, len(lines)):
        angle = float(lines[i][3])
        #if angle == 0 and np.random.uniform() <= 0.9:
        #    continue
        for j in range(3):
            if (lines[i][j][0] == ' '):
                lines[i][j] = lines[i][j][1:]
            if (lines[i][j][0] != '/'):
              lines[i][j] = '/' + lines[i][j]
           
            path = data_path  +  lines[i][j]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            measurements.append(float(lines[i][3]) + offset[j])  
    return images, measurements

def augment_images(images, measurements):
    length = len(images)
    for i in range(length):
        augmented_image = np.fliplr(images[i])
        augmented_measurement = -measurements[i]
        images.append(augmented_image)
        measurements.append(augmented_measurement)
        
lines = read_csv( "/root/Desktop/data3")
images, measurements = read_images("", lines)

dirs = [  "/root/Desktop/left_to_center" , "/root/Desktop/curve",   "/root/Desktop/bridge2", "/root/Desktop/curve2", #"/root/Desktop/curve2",
#     "/root/Desktop/curve4", "/root/Desktop/ccw2",
       ] 

for data_path in dirs:
    tmp_lines = read_csv(data_path)
    tmp_images, tmp_measurements = read_images("", tmp_lines)

    images.extend(tmp_images)
    measurements.extend(tmp_measurements)

augment_images(images, measurements)

X_train = np.array(images)
y_train = np.array(measurements)

print("num of sample = {}, steering=0 = {}".format(y_train.shape, np.where(y_train == 0.0)[0].size))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import plot_model
from keras.layers.core import Dropout
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.layers.advanced_activations import LeakyReLU

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=( (50, 20), (0, 0) )))


model.add(Conv2D( filters=24, kernel_size=(5, 5), strides=(2, 2)))
model.add(LeakyReLU())
#model.add(MaxPooling2D())
model.add(Conv2D( filters=36, kernel_size=(5, 5),  strides=(2, 2)))
model.add(LeakyReLU())
#model.add(MaxPooling2D())
model.add(Conv2D( filters=48, kernel_size=(5, 5),strides=(2, 2)))
model.add(LeakyReLU())
#model.add(MaxPooling2D())
model.add(Conv2D( 64, 3, 3))
model.add(LeakyReLU())
#model.add(MaxPooling2D())
model.add(Conv2D(64, 3, 3))
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(960))
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(LeakyReLU())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
plot_model(model, to_file='model.png')
model.summary()

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5')

# cut
# 23476/23476 [==============================] - 42s - loss: 0.1029 - val_loss: 0.0933
# cut, add ccw data
# 28756/28756 [==============================] - 51s - loss: 0.0950 - val_loss: 0.0916