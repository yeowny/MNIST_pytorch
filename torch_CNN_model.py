# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowny
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

import torch.nn as nn
import torch.nn.functional as F


class mnist_CNN_model(nn.Module):
    def __init__(self):
        super(mnist_CNN_model, self).__init__()
        self.model_path = 'torch_model'
        self.model_name = 'mnist_CNN_model'

        self.dropout_p = 0.5

        self.conv_layer = nn.Sequential(
            self.fn_conv_layer(1, 128, 3),
            self.fn_conv_layer(128, 64, 3),
            self.fn_conv_layer(64, 32, 3)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(32*19*19, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, 10)
        )

    def fn_conv_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 1),
            nn.Dropout(self.dropout_p)
        )

    def forward(self, x):
        conv_outp = self.conv_layer(x)
        # flatten_conv_outp = conv_outp.view((-1, np.prod(list(conv_outp.size())[1:])))
        flatten_conv_outp = conv_outp.view((-1, 32*19*19))
        fc_oupt = self.fc_layer(flatten_conv_outp)

        return F.log_softmax(fc_oupt)


if __name__ == '__main__':
    import torch
    from torch_load_mnist_data import load_mnist_data, show_mnist_data
    from torch_model_train_test import model_train_test
    import os

    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except NameError as e:
        print('{0}\n{1}\n{0}'.format('!'*50, e))
# =============================================================================
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# =============================================================================
    mnist_file_path = 'mnist_data'
# =============================================================================
    mnist_train, mnist_vali, mnist_test = load_mnist_data()
    print('%s\ntrain set : %s / %s' % ('#' * 100, mnist_train.__len__(), mnist_train.__getitem__(0)[0].size()))
    print('validation set : %s / %s' % (mnist_vali.__len__(), mnist_vali.__getitem__(0)[0].size()))
    print('test set : %s / %s\n%s\n' % (mnist_test.__len__(), mnist_test.__getitem__(0)[0].size(), '#' * 100))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = mnist_CNN_model().to(device)
    print(Model)
    Model_train_test = model_train_test(Model, device)

    Model_train_test.train(mnist_train, mnist_vali)
    test_y_dict = Model_train_test.test(mnist_test)
    model_oupt, pred_y = Model_train_test.predict(mnist_test.__getitem__(0)[0].numpy())

    print('=' * 100)
    show_mnist_data(mnist_test.__getitem__(0)[0])
    print('{}\n{}\n{}'.format(pred_y, [round(xx, 8) for xx in model_oupt], '=' * 100))

# ====================================================================================================
# mnist_CNN_model
#
# Test_loss = 0.0181
# Test_acc = 0.9940
# ====================================================================================================
