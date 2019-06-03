import numpy as np
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix

import sys


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

if len(sys.argv) < 3:
    print("Invalid argument!")
    print("Usage: {} TEST_SET MODEL_PATH".format(sys.argv[0]))
    exit()

test_folder = sys.argv[1]
model_folder = sys.argv[2]

json_file = open('{}/model_100.json'.format(model_folder), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("{}/model_100.h5".format(model_folder))

X_test = np.empty((0, 336))
Y_test = np.array([])

print('loading ' + test_folder)
X_test, Y_test = load_from_folder(X_test, Y_test, test_folder)

print('testing')
predictions = model.predict(X_test)

pred = [round(x[0]) for x in predictions]

acc_count = 0
for k in range(0,len(pred)):
    if pred[k] == Y_test[k]:
        acc_count += 1

cm = confusion_matrix(Y_test, pred)
print('confusion matrix:')
print cm

accuracy = float(acc_count)/len(pred)
print 'accuracy is : ' + str(accuracy)
