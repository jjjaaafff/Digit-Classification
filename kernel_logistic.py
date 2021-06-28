from scipy.io import loadmat
import scipy
import imageio
import numpy as np
import copy
import torch
import random


def rbf_kernel(X, Y, sigma):
    X_norm = np.linalg.norm(X, axis=1) ** 2
    Y_norm = np.linalg.norm(Y, axis=1) ** 2
    return np.exp(-0.5 / sigma ** 2 * (X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)))


def poly_kernel(X, Y, d):
    return np.dot(X.T, Y) ** d


def cosine_kernel(X, Y):
    X_norm = np.linalg.norm(X, axis=0)
    Y_norm = np.linalg.norm(Y, axis=0)
    return np.dot(X.T, Y) / (X_norm[:, None] * Y_norm[None, :])



class LogisticRegression_onebatch:
    def __init__(self, lr, datasize, kernel, premodel=None):
        self.category_num = 10
        self.lr = lr
        self.kernel = kernel
        if premodel is None:
            self.c_list = np.random.rand(self.category_num, datasize)
        else:
            self.c = np.load(premodel)

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))


    def train(self,data,label):
        lbda = 0.01
        d =2

        self.train_data = data

        batch_label = label.flatten()

        # K_batch_data = rbf_kernel(data.T, data.T, d)
        K_batch_data = poly_kernel(data, data, d)
        # K_batch_data = cosine_kernel(data, data)



        # K shape (samples, samples)

        bool_label = np.array([[1 if j == batch_label[i] % 10 else 0 for j in range(10)] for i in range(batch_label.shape[0])])
        # bool_label shape (samples, categories)


        # print("before",self.c_list[0][0])
        for current_category in range(self.category_num):
            current_c = self.c_list[current_category]
            projed_data = np.dot(K_batch_data,current_c)
            # projed_data shape (3000,)

            grad = np.dot(K_batch_data.T, 1 - 1 / (1 + np.exp(projed_data)) - bool_label[:,current_category]) + lbda * np.sign(current_c)
            self.c_list[current_category] -= self.lr * grad
        # print("after",self.c_list[0][0])


    def test(self,data,label):
        d = 2
        data_size = label.shape[0]
        data = data.reshape(1764, data_size)

        # y_pred = np.dot(rbf_kernel(data.T, self.train_data.T, d), self.c_list.T)
        y_pred = np.dot(poly_kernel(data, self.train_data, d), self.c_list.T)
        # y_pred = np.dot(cosine_kernel(data, train_data), self.c_list.T)


        # y_pred shape (test_size, categories)

        # print(y_pred[0])
        # print(y_pred[1])
        cnt = 0
        for i in range(data_size):
            if np.argmax(y_pred[i]) == label[i]%10:
                cnt+=1

        print(f'Accurate: {cnt}/{data_size}')
        acc = cnt/data_size
        return acc


if __name__ == '__main__':
    # parameter setting
    train_size = 3000
    test_size = 500



    # train_m = torch.load('../data/MNIST/processed/training.pt')
    # train_data = train_m[0][0:train_size,:,:]
    # train_data = train_data.permute(1,2,0).numpy().astype('float32')
    # # train data (28, 28, 6000)
    #
    # train_label = train_m[1][0:train_size]
    # train_label = train_label.numpy()
    # train_label = train_label[:, np.newaxis]
    # train_data = train_data/255
    #
    #
    # test_m = torch.load('../data/MNIST/processed/test.pt')
    # test_data = test_m[0][0:test_size,:,:]
    # test_data = test_data.permute(1,2,0).numpy().astype('float32')
    # # train data (28, 28, 6000)
    #
    # test_label = test_m[1][0:test_size]
    # test_label = test_label.numpy()
    # test_label = test_label[:, np.newaxis]
    # test_data = test_data/255


    # train_m = loadmat("./dataset/train_32x32.mat")
    # train_data = train_m["X"][:,:,:,0:train_size].astype('float32')
    # # train_data: (32, 32, 3, 73257)
    # train_data = train_data/255.
    #
    # train_label = train_m["y"][0:train_size,:]
    #
    #
    # test_m = loadmat("./dataset/test_32x32.mat")
    # test_data = test_m["X"][:,:,:,0:test_size].astype('float32')
    # # test_data: (32, 32, 3, 26032)
    # test_data = test_data/255.
    # test_label = test_m["y"][0:test_size,:]

    # Train dataset

    train_data = np.load("./dataset/hog_train.npy")
    train_data = train_data[:,0:train_size]
    train_m = loadmat("./dataset/train_32x32.mat")
    train_label = train_m["y"][0:train_size, :]

    # Test dataset
    test_data = np.load("./dataset/hog_test.npy")
    test_data = test_data[:,0:test_size]
    test_m = loadmat("./dataset/test_32x32.mat")
    test_label = test_m["y"][0:test_size, :]


    logisreg = LogisticRegression_onebatch(0.001,train_size,"rbf")

    print("train",train_data.shape)

    test_acc = []
    for i in range(300):
        logisreg.train(train_data,train_label)
        acc = logisreg.test(test_data,test_label)
        test_acc.append(acc)
    print(test_acc)




