import cv2
import numpy as np
import tqdm
from scipy.io import loadmat

if __name__ == '__main__':

    win_size = (32, 32)
    block_size = (8, 8)
    block_stride = (4, 4)
    cell_size = (4, 4)
    bin = 9
    HOG = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bin)

    # Extract Train HOG
    train_m = loadmat("./dataset/train_32x32.mat")
    train_size = train_m["X"].shape[-1]

    feature_size = ((win_size[0]-block_size[0])//block_stride[0] + 1)**2 * bin * 4
    res = np.zeros((feature_size,train_size))
    for i in tqdm.tqdm(range(train_size)):
        img = train_m["X"][:, :, :, i]
        feature = HOG.compute(img).flatten()
        res[:,i] = feature

    np.save("./dataset/hog_train.npy",res)
    print("Train HOG Feature Saved")

    # Extract Test HOG
    test_m = loadmat("./dataset/test_32x32.mat")
    test_size = test_m["X"].shape[-1]

    feature_size = ((win_size[0]-block_size[0])//block_stride[0] + 1)**2 * bin * 4
    res = np.zeros((feature_size,test_size))
    for i in tqdm.tqdm(range(test_size)):
        img = test_m["X"][:, :, :, i]
        feature = HOG.compute(img).flatten()
        res[:,i] = feature

    np.save("./dataset/hog_test.npy",res)
    print("Test HOG Feature Saved")






