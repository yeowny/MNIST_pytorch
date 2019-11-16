# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowny
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os, time, sys


class model_train_test:
    def __init__(self, Model, device):
        self.Model = Model
        self.device = device

    def _train_vali(self, bool_train, x_data, batch_size, optimizer, criterion, value_dict, bool_show=False):
        len_x = x_data.__len__()
        proceeding_bar_var = [0, 0]
        total_loss, total_acc = 0, 0
        for batch_x, batch_y in DataLoader(x_data, batch_size=batch_size, shuffle=bool_train):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            len_batch_x = batch_x.__len__()
            if bool_show:
                proceeding_bar_var = self._show_proceeding_bar(len_batch_x, len_x, proceeding_bar_var)

            if bool_train:
                optimizer.zero_grad()

            model_oupt = self.Model(batch_x)
            loss = criterion(model_oupt, batch_y)

            if bool_train:
                loss.backward()
                optimizer.step()

            acc = model_oupt.data.max(1)[1].eq(batch_y.data).sum()

            total_loss += loss.item() * len_batch_x
            total_acc += acc.item()

        value_dict['loss'].append(total_loss/len_x)
        value_dict['acc'].append(total_acc/len_x)

        return value_dict

    def _early_stopping_and_save_model(self, vali_loss_list, early_stopping_patience, bool_show_epoch):
        bool_save_check = False
        if len(vali_loss_list) > early_stopping_patience + 1:
            if vali_loss_list[-early_stopping_patience - 1] < min(vali_loss_list[-early_stopping_patience:]):
                return True
        try:
            if vali_loss_list[-1] == min(vali_loss_list):
                raise ValueError
        except ValueError:
            bool_save_check = True
            torch.save(self.Model.state_dict(), os.path.join(self.Model.model_path, '%s.pth' % self.Model.model_name))

        if bool_show_epoch:
            if bool_save_check:
                sys.stdout.write('*')
            sys.stdout.write('\n')
            sys.stdout.flush()

        return False

    def _show_proceeding_bar(self, len_batch_x_train, len_x_train, proceeding_bar_var):
        proceeding_bar_var[0] += len_batch_x_train
        proceeding_bar_print = int(proceeding_bar_var[0] * 100 / len_x_train) - proceeding_bar_var[1]
        if proceeding_bar_print != 0:
            sys.stdout.write('-' * proceeding_bar_print)
            sys.stdout.flush()

            proceeding_bar_var[1] += (int(proceeding_bar_var[0] * 100 / len_x_train) - proceeding_bar_var[1])

        return proceeding_bar_var

    def _show_training_graph(self, last_epoch, log_file_list, train_value_dict, vali_value_dict, bool_save_log_file):
        pd.DataFrame(log_file_list, columns=['epoch', 'train_loss', 'train_acc', 'vali_loss', 'vali_acc', 'time']) \
            .to_csv(os.path.join(self.Model.model_path, self.Model.model_name, '%s_log.csv' % self.Model.model_name)
                    , index=False)

        plt.clf()
        epoch_list = [i for i in range(1, last_epoch + 2)]
        graph_acc_list = [train_value_dict['acc'], vali_value_dict['acc'], 'r--', 'b--', 'acc']
        graph_loss_list = [train_value_dict['loss'], vali_value_dict['loss'], 'r', 'b', 'loss']
        for train_l_a_list, vali_l_a_list, trian_color, vali_color, loss_acc in [graph_acc_list, graph_loss_list]:
            plt.plot(epoch_list, train_l_a_list, trian_color, label='train_' + loss_acc)
            plt.plot(epoch_list, vali_l_a_list, vali_color, label='validation_' + loss_acc)
            plt.xlabel('epoch')
            plt.ylabel(loss_acc)
            plt.legend(loc='lower left')
            plt.title(self.Model.model_name)

        plt.savefig(os.path.join(self.Model.model_path, self.Model.model_name, '%s_plot.png' % self.Model.model_name))
        if bool_save_log_file:
            plt.show()

    def train(self, train_data, vali_data, epoch_num=1000, batch_size=512, optimizer=optim.Adam, learning_rate=1e-4,
              criterion=nn.CrossEntropyLoss(), early_stopping_patience=10, print_term=1,
              bool_show_proceeding_bar=True, bool_save_log_file=True, bool_show_img=True):
        for create_dir_name in [self.Model.model_path, os.path.join(self.Model.model_path, self.Model.model_name)]:
            try:
                os.mkdir(create_dir_name)
            except FileExistsError:
                pass

        log_file_list = []
        train_value_dict = {'loss': [], 'acc': []}
        vali_value_dict = {'loss': [], 'acc': []}
        optimizer = optimizer(self.Model.parameters(), lr=learning_rate)

        start_time = time.time()
        print('\n%s\n%s - training....' % ('-' * 100, self.Model.model_name))

        for epoch in range(epoch_num):
            bool_show_epoch = False
            if print_term != 0:
                if (epoch + 1) % print_term == 0:
                    bool_show_epoch = True

            self.Model.train()
            train_value_dict = self._train_vali(True, train_data, batch_size, optimizer, criterion, train_value_dict,
                                                bool_show=bool_show_proceeding_bar and bool_show_epoch)

            self.Model.eval()
            vali_value_dict = self._train_vali(False, vali_data, batch_size, optimizer, criterion, vali_value_dict)

            tmp_running_time = time.time() - start_time
            if bool_save_log_file:
                log_file_list.append([epoch + 1, train_value_dict['loss'][-1], train_value_dict['acc'][-1],
                                      vali_value_dict['loss'][-1], vali_value_dict['acc'][-1], tmp_running_time])
            if bool_show_epoch:
                if bool_show_proceeding_bar:
                    sys.stdout.write('\n')
                print('#%4d/%d' % (epoch + 1, epoch_num), end='  |  ')
                print('Train: loss=%.4f/acc=%.4f' % (train_value_dict['loss'][-1], train_value_dict['acc'][-1])
                      , end='  |  ')
                print('Validtion: loss=%.4f/acc=%.4f' % (vali_value_dict['loss'][-1], vali_value_dict['acc'][-1])
                      , end='  |  ')
                print('%sm %ss' % (str(int(tmp_running_time // 60)).zfill(2),
                                   ('%4.2f' % (tmp_running_time % 60)).zfill(5)), end='  |  ')

            if self._early_stopping_and_save_model(vali_value_dict['loss'], early_stopping_patience, bool_show_epoch):
                print('\n%s\nstop epoch : %d\n%s' % ('-' * 100, epoch - early_stopping_patience + 1, '-' * 100))
                break

        if bool_save_log_file:
            self._show_training_graph(epoch, log_file_list, train_value_dict, vali_value_dict, bool_show_img)

    def test(self, test_data, batch_size=512, criterion=nn.CrossEntropyLoss()):
        self.Model.load_state_dict(torch.load(os.path.join(self.Model.model_path, '%s.pth' % self.Model.model_name)))

        len_x_test = test_data.__len__()
        test_total_loss, test_total_acc = 0, 0
        test_y_dict = {'true': [], 'pred': []}

        self.Model.eval()
        for batch_x_test, batch_y_test in DataLoader(test_data, batch_size=batch_size):
            batch_x_test, batch_y_test = batch_x_test.to(self.device), batch_y_test.to(self.device)
            len_batch_x_test = batch_x_test.__len__()

            model_oupt = self.Model(batch_x_test)
            loss = criterion(model_oupt, batch_y_test)

            tmp_true_y = batch_y_test.data.tolist()
            tmp_pred_y = model_oupt.data.max(1)[1].tolist()

            test_y_dict['true'].extend(tmp_true_y)
            test_y_dict['pred'].extend(tmp_pred_y)
            acc = sum(np.equal(tmp_true_y, tmp_pred_y))

            test_total_loss += loss.item() * len_batch_x_test
            test_total_acc += acc

        test_loss = test_total_loss / len_x_test
        test_acc = test_total_acc / len_x_test

        print('\n' + '=' * 100)
        print(classification_report(test_y_dict['true'], test_y_dict['pred']))
        print(pd.crosstab(pd.Series(test_y_dict['true']), pd.Series(test_y_dict['pred']), rownames=['True'],
                          colnames=['Predict'], margins=True))
        print('\n%s\n%s\n' % ('=' * 100, self.Model.model_name))
        print('Test_loss = %.4f\nTest_acc = %.4f\n%s' % (test_loss, test_acc, '=' * 100))

        return test_y_dict

    def predict(self, one_data):
        self.Model.load_state_dict(torch.load(os.path.join(self.Model.model_path, '%s.pth' % self.Model.model_name)))

        self.Model.eval()
        model_oupt = F.softmax(self.Model(torch.from_numpy(np.asarray([one_data])).to(self.device))[0]).tolist()

        return model_oupt, np.argmax(model_oupt)
