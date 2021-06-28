from scipy.io import loadmat
import scipy
import imageio
import numpy as np
import copy
import random


class LogisticRegression:
    def __init__(self, lr, data_dimension, premodel=None, mode=None):
        self.category_num = 10
        self.lr = lr
        self.data_dim = data_dimension
        self.mode = mode
        if premodel is None:
            self.beta_list = np.ones((self.category_num, data_dimension+1))
        else:
            self.beta_list = np.load(premodel)

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))


    def train(self,data,label):

        lbda = 0.001

        data_size = label.shape[0]

        batch_data = np.row_stack((data, np.ones((1, data_size))))

        batch = [i for i in range(data_size)]
        random.shuffle(batch)

        if self.mode != None and self.mode!='Ridge' and self.mode!='Lasso':
            print(f"Only support Ridge and Lasso mode (default for None), but now we get {self.mode}")
            raise NotImplementedError


        # print("shape",batch_data.shape)
        for current_category in range(self.category_num):
            current_beta = self.beta_list[current_category]
            sign_beta = np.sign(current_beta)
            for j in batch:
                betaTx = np.dot(current_beta, batch_data[:,j])
                if label[j]%10 == current_category:
                    label_flag = 1
                else:
                    label_flag = 0

                if self.mode == "Ridge":
                    grad = batch_data[:, j] * (self.sigmoid(betaTx) - label_flag) + lbda * current_beta
                elif self.mode == "Lasso":
                    grad = batch_data[:, j] * (self.sigmoid(betaTx) - label_flag) + lbda * sign_beta
                else:
                    grad = batch_data[:, j] * (self.sigmoid(betaTx) - label_flag)

                self.beta_list[current_category] -= self.lr * grad


    def test(self,data,label):
        data_size = label.shape[0]
        cnt = 0
        image = data.reshape((self.data_dim,data_size))
        image = np.row_stack((image, np.ones((1, data_size))))

        pred_res = np.dot(self.beta_list,image).T
        for i in range(data_size):
            if np.argmax(pred_res[i]) == label[i]%10:
                cnt+=1
        acc = cnt/data_size
        print(f'Accurate: {cnt}/{data_size}  [{acc}]')
        return acc


if __name__ == '__main__':
    # parameter setting
    train_size = 10000
    test_size = 1000
    epoch = 100
    learn_rate = 0.001
    save_flag = False

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

    data_dimension = train_data.shape[0]

    logisreg = LogisticRegression(learn_rate, data_dimension, mode="Lasso")

    res_acc = []


    for i in range(epoch):
        logisreg.train(train_data,train_label)
        acc = logisreg.test(test_data,test_label)
        res_acc.append(acc)

    print(res_acc)

    if save_flag:
        np.save("./model/logistic_lasso.npy",logisreg.beta_list)













