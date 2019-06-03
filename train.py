import sys

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import numpy as np


def load_from_folder(X, Y, path):
    negative_count = int(open(path + '/negative/count.txt', 'r').read())
    positive_count = int(open(path + '/positive/count.txt', 'r').read())

    for i in range(negative_count):
        try:
            file_name = '{}/negative/{}.txt'.format(path, i)
            file_obj = open(file_name, 'r')
            vif = np.loadtxt(file_obj)
            vif = np.reshape(vif, (-1, vif.shape[0]))

            X = np.vstack((X, vif))
            Y = np.append(Y, 0)

            file_obj.close()
        except:
            print 'error in reading {}/negative/{}.txt'.format(path, i)

    for i in range(positive_count):
        try:
            file_name = '{}/positive/{}.txt'.format(path, i)
            file_obj = open(file_name, 'r')
            vif = np.loadtxt(file_obj)
            vif = np.reshape(vif, (-1, vif.shape[0]))

            X = np.vstack((X, vif))
            Y = np.append(Y, 1)

            file_obj.close()
        except:
            print 'error in reading {}/positive/{}.txt'.format(path, i)

    return X, Y

def build_model():
    model = Sequential()
    model.add(Dense(255, activation="relu", kernel_initializer="uniform", input_dim=336))

    for l in range(1,2):
        model.add(Dense(336, activation='relu', kernel_initializer="uniform"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if len(sys.argv) < 3:
    print("Invalid argument!")
    print("Usage: {} INPUT_1 INPUT_2 ... INPUT_N OUTPUT".format(sys.argv[0]))
    exit()

output_folder = sys.argv[-1]

X_train = np.empty((0, 336))
Y_train = np.array([])

for path in sys.argv[1:-1]:
    print('loading ' + path)
    X_train, Y_train = load_from_folder(X_train, Y_train, path)

print('training')
model = build_model()
model.fit(X_train, Y_train, epochs=150, batch_size=10, verbose=0)

# save model to disk
model_json = model.to_json()

with open("{}/model_100.json".format(output_folder), "w") as json_file:
    json_file.write(model_json)

model.save_weights("{}/model_100.h5".format(output_folder))
print("Saved model to disk")
