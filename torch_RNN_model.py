# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowy
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

import torch.nn as nn
import torch.nn.functional as F


class mnist_RNN_model(nn.Module):
    def __init__(self):
        super(mnist_RNN_model, self).__init__()
        self.model_path = 'torch_model'
        self.model_name = 'mnist_RNN_model'

        self.dropout_p = 0.5

        # self.rnn_layer = nn.RNN(input_size=28, hidden_size=512, num_layers=1, bias=True, dropout=self.dropout_p,
        #                         batch_first=True, bidirectional=True)
        # self.lstm_layer = nn.LSTM(input_size=28, hidden_size=512, num_layers=1, bias=True, dropout=self.dropout_p,
        #                         batch_first=True, bidirectional=True)
        self.gru_layer = nn.GRU(input_size=28, hidden_size=512, num_layers=2, bias=True, dropout=self.dropout_p,
                                batch_first=True, bidirectional=True)

        self.fc_layer = nn.Sequential(
            nn.Linear(512*2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        re_x = x.view((-1, 28, 28))
        # rnn_outp, h_n = self.gru_layer(re_x)
        # lstm_outp, (h_n, c_n) = self.gru_layer(re_x)
        gru_outp, h_n = self.gru_layer(re_x)

        fc_oupt = self.fc_layer(gru_outp[:, -1, :])

        return F.log_softmax(fc_oupt)


if __name__ == '__main__':
    import torch
    from torch_load_mnist_data import load_mnist_data
    from torch_model_train import model_train
    from torch_model_test import model_test
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
    print('%s\ntrain set : %s / %s' % ('#' * 100, mnist_train.__len__(), list(mnist_train.__getitem__(0)[0].size())))
    print('validation set : %s / %s' % (mnist_vali.__len__(), list(mnist_vali.__getitem__(0)[0].size())[1:]))
    print('test set : %s / %s\n%s\n' % (mnist_test.__len__(), list(mnist_test.__getitem__(0)[0].size())[1:], '#' * 100))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(mnist_RNN_model())
    Model = mnist_RNN_model().to(device)
    model_train(Model, device, mnist_train, mnist_vali)
    model_test(Model, device, mnist_test)

# ====================================================================================================
# mnist_RNN_model
#
# Test_loss = 0.0294
# Test_acc = 0.9913
# ====================================================================================================
