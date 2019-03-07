import numpy as np
from core.video import Video

def load_features(path):
    positive_file = open("{}/positive.txt".format(path))
    negative_file = open("{}/negative.txt".format(path))
    data = []
    target = []

    for line in positive_file:
        words = line.replace('[', '').replace(']', '').split()
        vector = [1]
        for word in words:
            vector.append(int(word))

        data.append(vector)

    for line in negative_file:
        words = line.replace('[', '').replace(']', '').split()
        vector = [0]
        for word in words:
            vector.append(int(word))

        data.append(vector)

    data = np.array(data)
    np.random.shuffle(data)

    return data[:,1:], data[:,0]
