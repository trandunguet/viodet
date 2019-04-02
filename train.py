#!/usr/bin/env python3
import sys

from sklearn import datasets, svm, metrics
import numpy as np

import core

if __name__ == '__main__':
      if len(sys.argv) != 2:
            print("Invalid argument!")
            print("Usage: {} FEATURES_FOLDER_PATH".format(sys.argv[0]))
            exit()

      data, target = core.load_features(sys.argv[1])
      n_samples = len(data)
      alpha = 1 / 2

      print("loaded {} samples".format(n_samples))

      print("data shape: {} \ntarget shape: {}".format(data.shape, target.shape))

      # Create a classifier: a support vector classifier
      classifier = svm.SVC(gamma='auto')

      print("training")

      # We learn the digits on the first half of the digits
      classifier.fit(data[:int(n_samples * alpha)], target[:int(n_samples * alpha)])

      print("predicting")

      # Now predict the value of the digit on the second half:
      expected = target[int(n_samples * (1 - alpha)):]
      predicted = classifier.predict(data[int(n_samples * (1 - alpha)):])

      print("Classification report for classifier %s:\n%s\n"
            % (classifier, metrics.classification_report(expected, predicted)))
      print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
