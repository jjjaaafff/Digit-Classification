from scipy.io import loadmat
import scipy
import imageio
import numpy as np
import copy


class LDA:
    def __init__(self,out_dim,premodel=None):
        self.category_num = 10
        self.out_dim = out_dim
        if premodel is None:
            self.w = None
        else:
            self.w = np.load(premodel)

    def train(self,data,origin_label):
        # data: (1764, n_samples)
        label = origin_label
        # label: (n_samples, 1)
        total_mean = np.mean(data,1)
        # total_mean: (1764,), avg of all samples

        data_size = data.shape[-1]

        category_mean = np.zeros((self.category_num,data.shape[0]))
        # avg of each category
        label_cnt = []
        for i in range(self.category_num):
            idx = np.where(label%10 == i)
            label_cnt.append(idx[0])
            for tmp in idx[0]:
                category_mean[i] += data[:,tmp]
            if len(idx[0]) != 0:
                category_mean[i] /= len(idx[0])

        mean_difference = category_mean - total_mean
        # mean_difference: (3072, )
        weighted_mean_difference = copy.deepcopy(mean_difference)
        for i in range(self.category_num):
            weighted_mean_difference[i] *= len(label_cnt[i])


        Sb = np.matmul(weighted_mean_difference.T,mean_difference)
        # Sb: (3072, 3072)

        Sw = np.zeros_like(Sb)
        # Sw: (3072, 3072)

        for i in range(data.shape[1]):
            std = data[:,i] - category_mean[label[i][0]%10]
            std = std[:, np.newaxis]
            Sw += np.matmul(std, std.T)

        # Calculate the eigenvalue and eigenvector of Sw^(-1)Sb
        SwSb = np.matmul(np.linalg.pinv(Sw), Sb)
        eig_value, eig_vector = np.linalg.eig(SwSb)
        sorted_eigvalue_idx = sorted(range(len(eig_value)), key=lambda k: eig_value[k], reverse=True)

        # choose the top out_dim columns in eigenvector matrix with highest eigenvalue
        feature_vector = np.array([eig_vector[:, i] for i in sorted_eigvalue_idx[:self.out_dim]])
        # feature_vector: (out_dim, 3072)

        self.w = feature_vector
        np.save("./model/LDAfeature.npy", feature_vector)

        projected_data = np.matmul(self.w, data).T
        self.gauss_mean = []
        self.gauss_conv = []
        self.gauss_para = []
        for i in range(self.category_num):
            idx = np.where(label % 10 == i)
            i_gauss_mean = np.mean(projected_data[idx[0]], axis=0)
            # i_gauss_mean (out_dim, )

            i_gauss_conv = np.cov(projected_data[idx[0]], rowvar=False)
            # i_gauss_conv (out_dim, out_dim)

            i_weight = len(idx[0]) / data_size
            gauss_parameter = 1/((2*np.pi)**(self.out_dim/2) * np.linalg.det(i_gauss_conv)**(0.5))

            self.gauss_mean.append(i_gauss_mean)
            self.gauss_conv.append(i_gauss_conv)
            self.gauss_para.append(i_weight * gauss_parameter)
        return

    def test(self, data, label):
        data_size = data.shape[-1]
        projected_data = np.matmul(self.w, data).T
        # projected_data: (n_samples, out_dim)

        # Calculate the n-normalization distribution of each category
        likelyhood = []
        for i in range(self.category_num):
            gauss_pdf = self.gauss_para[i] * np.exp(-0.5* np.sum(
                np.matmul((projected_data-self.gauss_mean[i]),np.linalg.inv(self.gauss_conv[i]))*(projected_data-self.gauss_mean[i]),1))
            # gauss_pdf (n_samples,)

            likelyhood.append(gauss_pdf)


        likelyhood = np.array(likelyhood)
        # likelyhood (categoty, n_samples)

        predict = np.argmax(likelyhood,0)
        cnt = 0
        for i in range(data_size):
            if predict[i]%10 == label[i]:
                cnt += 1
        return cnt/data_size


if __name__ == '__main__':
    # parameter setting
    train_size = 20000
    test_size = 5000
    LDA_reduced_dim = 20


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



    # lda = LDA(LDA_reduced_dim,"./model/weight.npy")
    lda = LDA(LDA_reduced_dim)


    lda.train(train_data,train_label)

    test_acc = lda.test(test_data,test_label)
    print("test accuracy is ",test_acc)

    lala = lda.test(train_data,train_label)
    print("train accuracy is ",lala)




