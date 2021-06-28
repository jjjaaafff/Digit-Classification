from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.optim.lr_scheduler import StepLR
import imageio
from scipy.io import loadmat


class DataSet:
    def __init__(self, root, transform):
        train_m = loadmat(root)
        self.data = train_m["X"]
        self.label = train_m["y"]
        self.categories = 10
        self.transform = transform

    def num_classes(self):
        return self.categories

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        image = self.data[:, :, :, index]
        category = self.label[index,:][0] - 1

        if self.transform:
            image = self.transform(image) #image_shape torch.Size([3, 32, 32])
        return image, category

#nn即为神经网络，一个 nn.Module 包含若干 layers, 和一个方法 forward(input), 该方法返回 output
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #四个参数为：通道数，输出深度，filter的高，filter的宽
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)


        self.pool = nn.MaxPool2d(2, 2)
        #随机将整个通道置零，提升特征图之间的独立性
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        #线性变换，两个参数为：输入大小，输出大小
        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        #x的初始规模为：64组数据，3个通道，高32，宽32.即[64,3,32,32]

        # print("x",x.shape)

        x = self.conv1(x)  #[64,16,30,30]
        x = self.bn1(x)
        x = self.relu1(x)
        #x = F.relu(x)

        x = self.conv2(x)  #[64,32,28,28]
        x = self.bn2(x)
        x = self.relu2(x)
        #x = F.relu(x)
        x = self.pool(x)

        #提取重要信息，去掉不重要的信息，减少计算开销。[64,32,14,14]
        x = self.dropout1(x)
        x = torch.flatten(x, 1) #将纬度推平合并, [64,6272]
        x = self.fc1(x)  #[64,512]
        x = self.relu3(x)
        #x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  #[64,10]
        output = F.log_softmax(x, dim=1) #[64,10]
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #在每一次训练回环中，先将梯度缓存置零
        optimizer.zero_grad()

        output = model(data)

        # print("output",output)
        # print("target",target)

        #损失函数接受 (output, target) 作为输入，然后计算一个估计网络输出离我们的期望输出还有多远的评估值
        loss = criterion(output, target.long())

        #反向传播误差
        loss.backward()

        #执行更新，更新权重
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss /= (batch_idx + 1)
    train_acc = correct / total

    print('Epoch [%d] Loss: %.3f | Traininig Acc: %.3f%% (%d/%d)'
          % (epoch, train_loss,
             100. * train_acc, correct, total))

    return train_loss, train_acc



def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            c = (predicted == target).squeeze()
            for i in range(target.shape[0]):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    test_acc = correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * test_acc))
    return test_acc



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()



    torch.manual_seed(args.seed)

    #cuda可用时，可将张量在CPU和GPU之间移动，否则设备为CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


    dataset1 = DataSet(root="./dataset/train_32x32.mat", transform = transform)
    dataset2 = DataSet(root="./dataset/test_32x32.mat", transform = transform)


    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)

    #Torch.optim是一个实现各种优化算法的模块，用于构建神经网络
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)

    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)

    train_loss = []
    train_acc = []
    test_acc = []

    test(model, device, test_loader)
    for epoch in range(1, args.epochs + 1):
        a,b = train(args, model, device, train_loader, optimizer, epoch)
        train_loss.append(a)
        train_acc.append(b)
        c = test(model, device, test_loader)
        test_acc.append(c)
        scheduler.step()

    print("Train loss", train_loss)
    print("Train acc", train_acc)
    print("Test acc", test_acc)

    if args.save_model:
        torch.save(model, "cifar_model.pkl")




if __name__ == '__main__':
    main()