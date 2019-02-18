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

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')

# initial state with: epoch 7 6428/6428 [==============================] - 3s 518us/step - loss: 1599.1930 - val_loss: 1615.7068
# with normalization ( /255.0 - 0.5): 6428/6428 [==============================] - 8s 1ms/step - loss: 3.9704 - val_loss: 1.6130
# epoch = 2: 6428/6428 [==============================] - 8s 1ms/step - loss: 1.8163 - val_loss: 3.0627
