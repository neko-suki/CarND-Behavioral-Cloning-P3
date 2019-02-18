import csv
import  cv2
import numpy as np

def read_csv():
    lines = []
    with open("data/driving_log.csv", "r") as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines

def read_images(lines):
    images = []
    measurements = []
    for i in range(1, len(lines)):
        data_path = "./data"
        path = data_path +"/" +  lines[i][0]
        img = cv2.imread(path)
        images.append(img)
        measurements.append(lines[i][3])  

    return images, measurements

# ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
lines = read_csv()
images, measurements = read_images(lines)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D,MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 2,2, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 2, 2, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')

# lenet5, 2 epoch : 6428/6428 [==============================] - 13s - loss: 0.0105 - val_loss: 0.0117
# lenet reference https://engmrk.com/lenet-5-a-classic-cnn-architecture/
