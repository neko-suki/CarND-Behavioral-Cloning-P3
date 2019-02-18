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
        for j in range(3):
            if (lines[i][j][0] == ' '):
                lines[i][j] = lines[i][j][1:]
            path = data_path  + "/" +  lines[i][j]
            img = cv2.imread(path)
            images.append(img)
            measurements.append(float(lines[i][3]))  
    return images, measurements

def augment_images(images, measurements):
    length = len(images)
    for i in range(length):
        augmented_image = np.fliplr(images[i])
        augmented_measurement = -measurements[i]
        images.append(augmented_image)
        measurements.append(augmented_measurement)
        
# ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
lines = read_csv()
images, measurements = read_images(lines)
augment_images(images, measurements)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D,MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=( (70, 25), (0, 0) )))
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

# crop,  epoch2 : 38572/38572 [==============================] - 49s - loss: 0.0107 - val_loss: 0.0093

