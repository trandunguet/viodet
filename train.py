import sys
import random
import time

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import numpy as np

if len(sys.argv) != 2:
    print("Invalid argument!")
    print("Usage: {} INPUT_FOLDER".format(sys.argv[0]))
    exit()

input_folder = sys.argv[1]
X_train = np.empty((0,336))
Y_train = np.array([])

# load train negative
print('loading train negative')
data_negative = range(1000)
random.shuffle(data_negative)
for i in data_negative[:300]:
    try:
        file_name = '{}/negative/{}.txt'.format(input_folder, i)
        file_obj = open(file_name, 'r')
        vif = np.loadtxt(file_obj)
        vif = np.reshape(vif, (-1, vif.shape[0]))

        X_train = np.vstack((X_train,vif))
        Y_train = np.append(Y_train,0)

        file_obj.close()
    except:
        print 'error in reading negative/{}.txt'.format(i)

# load train positive
print('loading train positive')
data_postive = range(63)
random.shuffle(data_postive)
for i in data_postive[:50]:
    try:
        file_name = '{}/positive/{}.txt'.format(input_folder, i)
        file_obj = open(file_name, 'r')
        vif = np.loadtxt(file_obj)
        vif = np.reshape(vif, (-1, vif.shape[0]))

        X_train = np.vstack((X_train, vif))
        Y_train = np.append(Y_train, 1)

        file_obj.close()
    except:
        print 'error in reading positive/{}.txt'.format(i)

# train
print('training')
model = Sequential()
model.add(Dense(255, activation="relu", kernel_initializer="uniform", input_dim=336))

for l in range(1,2):
    model.add(Dense(336, activation='relu', kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=150, batch_size=10,  verbose=0)

print 'model trained'

# load test negative
print('loading test negative')
X_test = np.empty((0,336))
Y_test = np.array([])
for i in data_negative[300:400]:
    try:
        file_name = '{}/negative/{}.txt'.format(input_folder, i)
        file_obj = open(file_name, 'r')
        vif = np.loadtxt(file_obj)
        vif = np.reshape(vif, (-1, vif.shape[0]))

        X_test = np.vstack((X_test,vif))
        Y_test = np.append(Y_test,0)

        file_obj.close()
    except:
        print 'error in reading negative/{}.txt'.format(i)

# load test positive
print('loading test positive')
for i in data_postive[50:]:
    try:
        file_name = '{}/positive/{}.txt'.format(input_folder, i)
        file_obj = open(file_name, 'r')
        vif = np.loadtxt(file_obj)
        vif = np.reshape(vif, (-1, vif.shape[0]))

        X_test = np.vstack((X_test, vif))
        Y_test = np.append(Y_test, 1)

        file_obj.close()
    except:
        print 'error in reading positive/{}.txt'.format(i)

# test
print('testing')
predictions = model.predict(X_test)

pred = [round(x[0]) for x in predictions]

acc_count = 0
for k in range(0,len(pred)):
    if pred[k] == Y_test[k]:
        acc_count += 1

cm = confusion_matrix(Y_test, pred)
print cm

accuracy = float(acc_count)/len(pred)
print 'accuracy is : ' + str(accuracy)

# save model to disk
model_json = model.to_json()

with open("model_100.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_100.h5")
print("Saved model to disk")

