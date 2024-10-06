from modules.DHA_RNNs import dha_rnns

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from os.path import join as opj

# lstm or rnn or gru
epochs = 200
num_object = 5
batch_size = 2
hidden_size = 128
input_size = 128
num_layers = 1
learning_rate = 0.001
llm_data_path = r'data/llm'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net_list = ['lstm', 'rnn', 'gru']


def get_data_loader(dataset, batch_size, shuffle=False):
    dataset_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=False)
    return dataset_loader


train_dataset = torch.load(opj(llm_data_path, 'llm_train_dataset.pth'),
                           map_location=device)
test_dataset = torch.load(opj(llm_data_path, 'llm_test_dataset.pth'),
                          map_location=device)

train_loader = get_data_loader(train_dataset, batch_size=batch_size)
test_loader = get_data_loader(test_dataset, batch_size=1)

logging.getLogger().setLevel(logging.INFO)
for net_name in net_list:
    model = dha_rnns(hidden_size, hidden_size, num_layers, num_object, net_name)
    model.model_train(train_loader, epochs, learning_rate, device)

    # model.save('params/HOT_' + net_name.upper() + '.params')
    # model.load('params/HOT_' + net_name.upper() + '.params')
    auc, acc = model.model_test(test_loader, device)
    print('[' + net_name.upper() + ' Test] AUC:%.3f, ACC:%.3f' % (auc, acc))
