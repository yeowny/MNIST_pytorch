# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowny
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

import torch.nn as nn
import torch.nn.functional as F


class mnist_DNN_model(nn.Module):
    def __init__(self):
        super(mnist_DNN_model, self).__init__()
        self.model_path = 'torch_model'
        self.model_name = 'mnist_DNN_model'

        self.dropout_p = 0.5

        self.sequential_1 = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_p)
        )

        self.fc_2 = nn.Linear(512, 256)
        self.bn_2 = nn.BatchNorm1d(256)
        self.d_3 = nn.Dropout(self.dropout_p)

        self.fc_3 = nn.Linear(256, 128)
        self.bn_3 = nn.BatchNorm1d(128)
        self.d_3 = nn.Dropout(self.dropout_p)

        self.fc_oupt = nn.Linear(128, 10)

    def forward(self, x):
        re_x = x.view((-1, 28*28))

        fc_1 = self.sequential_1(re_x)

        fc_2 = self.fc_2(fc_1)
        bn_2 = F.relu(self.bn_2(fc_2))
        d_2 = self.d_3(bn_2)

        fc_3 = self.fc_3(d_2)
        bn_3 = F.elu(self.bn_3(fc_3))
        d_3 = self.d_3(bn_3)

        fc_oupt = self.fc_oupt(d_3)

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

    Model = mnist_DNN_model().to(device)
    print(Model)
    Model_train_test = model_train_test(Model, device)

    Model_train_test.train(mnist_train, mnist_vali)
    test_y_dict = Model_train_test.test(mnist_test)
    model_oupt, pred_y = Model_train_test.predict(mnist_test.__getitem__(0)[0].numpy())

    print('=' * 100)
    show_mnist_data(mnist_test.__getitem__(0)[0])
    print('{}\n{}\n{}'.format(pred_y, [round(xx, 8) for xx in model_oupt], '=' * 100))

# ====================================================================================================
# mnist_DNN_model
#
# Test_loss = 0.0588
# Test_acc = 0.9837
# ====================================================================================================
