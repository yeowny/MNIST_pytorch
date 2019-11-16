# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowny
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def load_mnist_data(mnist_data_file_path='./mnist_data/', random_seed=1234, validation_ratio=1/6):
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train_vali = datasets.MNIST(root=mnist_data_file_path, train=True, transform=mnist_transform,
                                      target_transform=None, download=True)
    mnist_test = datasets.MNIST(root=mnist_data_file_path, train=False, transform=mnist_transform,
                                target_transform=None, download=True)

    num_train_vali = mnist_train_vali.__len__()
    indices = list(range(num_train_vali))
    num_vali = int(validation_ratio * num_train_vali)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[num_vali:])
    valid_sampler = SubsetRandomSampler(indices[:num_vali])

    mnist_train = TensorDataset(*list(DataLoader(mnist_train_vali, num_train_vali-num_vali, sampler=train_sampler))[0])
    mnist_vali = TensorDataset(*list(DataLoader(mnist_train_vali, num_vali, sampler=valid_sampler))[0])
    mnist_test = TensorDataset(*list(DataLoader(mnist_test, mnist_test.__len__()))[0])

    return mnist_train, mnist_vali, mnist_test


def show_mnist_data(x_data_sample):
    plt.clf()
    plt.imshow(x_data_sample.reshape(28, 28), cmap='gray')
    plt.show()


if __name__ == '__main__':
    import os

    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except NameError as e:
        print('{0}\n{1}\n{0}'.format('!'*50, e))

    mnist_train, mnist_vali, mnist_test = load_mnist_data()
    print('%s\ntrain set : %s / %s' % ('#'*100, mnist_train.__len__(), list(mnist_train.__getitem__(0)[0].size())[1:]))
    print('validation set : %s / %s' % (mnist_vali.__len__(), list(mnist_vali.__getitem__(0)[0].size())[1:]))
    print('test set : %s / %s\n%s' % (mnist_test.__len__(), list(mnist_test.__getitem__(0)[0].size())[1:], '#'*100))

    show_mnist_data(mnist_train.__getitem__(0)[0])
