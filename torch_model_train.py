# -*- coding: utf-8 -*-

# =============================================================================
# @author: yeowy
# woon young, YEO
# ywy317391@gmail.com
# https://github.com/yeowny
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os, time, sys


def model_train(Model, device, train_data, vali_data, epoch_num=1000, batch_size=512, learning_rate=1e-4,
                early_stopping_patience=10, print_term=1, boolen_show_proceeding_bar=True, boolen_save_log_file=True):
    def early_stopping_and_save_model(vali_loss_list, early_stopping_patience, boolen_show_epoch):
        boolen_save_check = False
        if len(vali_loss_list) > early_stopping_patience + 1:
            if vali_loss_list[-early_stopping_patience - 1] < min(vali_loss_list[-early_stopping_patience:]):
                return True
        try:
            if vali_loss_list[-1] == min(vali_loss_list):
                raise ValueError
        except ValueError:
            boolen_save_check = True
            torch.save(Model.state_dict(), os.path.join(Model.model_path, '%s.pth' % Model.model_name))

        if boolen_show_epoch:
            if boolen_save_check:
                sys.stdout.write('*')
            sys.stdout.write('\n')
            sys.stdout.flush()
        return False

    def show_proceeding_bar(len_batch_x_train, len_x_train, proceeding_bar_var):
        proceeding_bar_var[0] += len_batch_x_train
        proceeding_bar_print = int(proceeding_bar_var[0] * 100 / len_x_train) - proceeding_bar_var[1]
        if proceeding_bar_print != 0:
            sys.stdout.write('-' * proceeding_bar_print)
            sys.stdout.flush()

            proceeding_bar_var[1] += (int(proceeding_bar_var[0] * 100 / len_x_train) - proceeding_bar_var[1])

        return proceeding_bar_var

    for create_dir_name in [Model.model_path, os.path.join(Model.model_path, Model.model_name)]:
        try:
            os.mkdir(create_dir_name)
        except FileExistsError:
            pass

    optimizer = optim.Adam(Model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    log_file_list = []
    len_x_train, len_x_vali = train_data.__len__(), vali_data.__len__()
    train_loss_list, vali_loss_list = [], []
    train_acc_list, vali_acc_list = [], []

    start_time = time.time()
    print('\n%s\n%s - training....' % ('-' * 100, Model.model_name))

    for epoch in range(epoch_num):
        boolen_show_epoch = False
        if print_term != 0:
            if (epoch + 1) % print_term == 0:
                boolen_show_epoch = True

        proceeding_bar_var = [0, 0]
        total_loss, total_acc, vali_total_loss, vali_total_acc = 0, 0, 0, 0

        Model.train()
        for batch_x_train, batch_y_train in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            batch_x_train, batch_y_train = batch_x_train.to(device), batch_y_train.to(device)
            len_batch_x_train = batch_x_train.__len__()
            if boolen_show_proceeding_bar and boolen_show_epoch:
                proceeding_bar_var = show_proceeding_bar(len_batch_x_train, len_x_train, proceeding_bar_var)

            optimizer.zero_grad()
            model_oupt = Model(batch_x_train)
            loss = criterion(model_oupt, batch_y_train)
            loss.backward()
            optimizer.step()

            acc = model_oupt.data.max(1)[1].eq(batch_y_train.data).sum()

            total_loss += loss.item() * len_batch_x_train
            total_acc += acc.item()

        train_loss_list.append(total_loss / len_x_train)
        train_acc_list.append(total_acc / len_x_train)

        Model.eval()
        for batch_x_vali, batch_y_vali in DataLoader(vali_data, batch_size=batch_size):
            batch_x_vali, batch_y_vali = batch_x_vali.to(device), batch_y_vali.to(device)
            len_batch_x_vali = batch_x_vali.__len__()

            model_oupt = Model(batch_x_vali)
            loss = criterion(model_oupt, batch_y_vali)

            acc = model_oupt.data.max(1)[1].eq(batch_y_vali.data).sum()

            vali_total_loss += loss.item() * len_batch_x_vali
            vali_total_acc += acc.item()

        vali_loss_list.append(vali_total_loss / len_x_vali)
        vali_acc_list.append(vali_total_acc / len_x_vali)

        # Model.eval()
        # for batch_x_vali, batch_y_vali in DataLoader(vali_data, batch_size=batch_size):
        #     batch_x_vali, batch_y_vali = batch_x_vali.to(device), batch_y_vali.to(device)
        #     len_batch_x_vali = batch_x_vali.__len__()
        #
        #     model_oupt = Model(batch_x_vali)
        #     loss = criterion(model_oupt, batch_y_vali)
        #
        #     acc = model_oupt.data.max(1)[1].eq(batch_y_vali.data).sum()
        #
        #     vali_total_loss += loss.item() * len_batch_x_vali
        #     vali_total_acc += acc.item()
        #
        # vali_loss_list.append(vali_total_loss / len_x_vali)
        # vali_acc_list.append(vali_total_acc / len_x_vali)

        tmp_running_time = time.time() - start_time
        if boolen_save_log_file:
            log_file_list.append([epoch + 1, train_loss_list[-1], train_acc_list[-1],
                                  vali_loss_list[-1], vali_acc_list[-1], tmp_running_time])
        if boolen_show_epoch:
            if boolen_show_proceeding_bar:
                sys.stdout.write('\n')
            print('#%4d/%d' % (epoch + 1, epoch_num), end='  |  ')
            print('Train: loss=%.4f/acc=%.4f' % (train_loss_list[-1], train_acc_list[-1]), end='  |  ')
            print('Validtion: loss=%.4f/acc=%.4f' % (vali_loss_list[-1], vali_acc_list[-1]), end='  |  ')
            print('%sm %ss' % (str(int(tmp_running_time // 60)).zfill(2),
                               ('%4.2f' % (tmp_running_time % 60)).zfill(5)), end='  |  ')

        if early_stopping_and_save_model(vali_loss_list, early_stopping_patience, boolen_show_epoch):
            print('\n%s\nstop epoch : %d\n%s' % ('-' * 100, epoch - early_stopping_patience + 1, '-' * 100))
            break

    if boolen_save_log_file:
        pd.DataFrame(log_file_list, columns=['epoch', 'train_loss', 'train_acc', 'vali_loss', 'vali_acc', 'time']) \
            .to_csv(os.path.join(Model.model_path, Model.model_name, '%s_log.csv' % Model.model_name), index=False)

        plt.clf()
        epoch_list = [i for i in range(1, epoch + 2)]
        graph_acc_list = [train_acc_list, vali_acc_list, 'r--', 'b--', 'acc']
        graph_loss_list = [train_loss_list, vali_loss_list, 'r', 'b', 'loss']
        for train_l_a_list, vali_l_a_list, trian_color, vali_color, loss_acc in [graph_acc_list, graph_loss_list]:
            plt.plot(epoch_list, train_l_a_list, trian_color, label='train_' + loss_acc)
            plt.plot(epoch_list, vali_l_a_list, vali_color, label='validation_' + loss_acc)
            plt.xlabel('epoch')
            plt.ylabel(loss_acc)
            plt.legend(loc='lower left')
            plt.title(Model.model_name)

        plt.savefig(os.path.join(Model.model_path, Model.model_name, '%s_plot.png' % Model.model_name))
        plt.show()
