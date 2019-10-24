# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowy
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import os


def model_test(Model, device, test_data, batch_size=512):
    criterion = nn.CrossEntropyLoss()

    len_x_test = test_data.__len__()
    test_total_loss, test_total_acc = 0, 0
    test_true_y, test_pred_y = [], []

    Model.load_state_dict(torch.load(os.path.join(Model.model_path, '%s.pth' % Model.model_name)))

    Model.eval()
    for batch_x_test, batch_y_test in DataLoader(test_data, batch_size=batch_size):
        batch_x_test, batch_y_test = batch_x_test.to(device), batch_y_test.to(device)
        len_batch_x_vali = batch_x_test.__len__()

        model_oupt = Model(batch_x_test)
        loss = criterion(model_oupt, batch_y_test)

        tmp_true_y = batch_y_test.data.tolist()
        tmp_pred_y = model_oupt.data.max(1)[1].tolist()

        test_true_y.extend(tmp_true_y)
        test_pred_y.extend(tmp_pred_y)
        acc = sum(np.equal(tmp_true_y, tmp_pred_y))

        test_total_loss += loss.item() * len_batch_x_vali
        test_total_acc += acc

    test_loss = test_total_loss / len_x_test
    test_acc = test_total_acc / len_x_test

    print('\n' + '=' * 100)
    print(classification_report(test_true_y, test_pred_y, target_names=['mnist_' + str(i) for i in range(10)]))
    print(pd.crosstab(pd.Series(test_true_y), pd.Series(test_pred_y), rownames=['True'], colnames=['Predicted'],
                      margins=True))
    print('\n%s\n%s\n' % ('=' * 100, Model.model_name))
    print('Test_loss = %.4f\nTest_acc = %.4f\n%s' % (test_loss, test_acc, '=' * 100))
