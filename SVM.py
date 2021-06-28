import numpy as np
from sklearn import svm
from scipy.io import loadmat
import copy
import torch
from utils import *


if __name__ == '__main__':
    # parameter setting
    train_size = 10000
    test_size = 1000
    linear_svm_flag = 1 # 1 for use linear svm, 0 for use kernel
    kernel_type = "rbf" # "linear" "sigmoid" "poly" "rbf"

    train_data = np.load("./dataset/hog_train_data.npy")
    train_data = train_data[:,0:train_size]
    train_data = train_data.T

    test_data = np.load("./dataset/hog_test_data.npy")
    test_data = test_data[:,0:test_size]
    test_data = test_data.T

    train_m = loadmat("./dataset/train_32x32.mat")
    train_label = train_m["y"][0:train_size,:]
    train_label = np.array(train_label).flatten()

    test_m = loadmat("./dataset/test_32x32.mat")
    test_label = test_m["y"][0:test_size,:]
    test_label = np.array(test_label).flatten()


    if linear_svm_flag == 1:
        print("Use Linear SVM")

        model = svm.LinearSVC()
        model.fit(train_data,train_label)

        train_pred_label = model.predict(train_data)
        train_acc = np.sum(train_pred_label == train_label) / train_size
        print("train acc ",train_acc)

        test_pred_label = model.predict(test_data)
        test_acc = np.sum(test_pred_label == test_label) / test_size
        print("test",test_acc)

    else:
        print(f"Use Kernel SVM with kernel {kernel_type}")

        model = svm.SVC(kernel=kernel_type)
        model.fit(train_data,train_label)

        train_pred_label = model.predict(train_data)
        train_acc = np.sum(train_pred_label == train_label) / train_size
        print("train acc ",train_acc)

        test_pred_label = model.predict(test_data)
        test_acc = np.sum(test_pred_label == test_label) / test_size
        print("test",test_acc)








