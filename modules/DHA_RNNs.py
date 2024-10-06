import logging
import torch
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score
from layers.RNNs import RNN_Net
from layers.RNNs import LSTM_Net
from layers.RNNs import GRU_Net
import pandas as pd
from utils.tools import get_acc, get_corrects


def process_pred(raw_pred, raw_true, object_num: int) -> tuple:
    objects = raw_true.flatten().to(torch.int64) % object_num
    length = objects.shape[0]
    preds = raw_pred[:length]
    preds = preds.gather(1, objects.view(-1, 1)).flatten()[:-1]
    targets = raw_true.flatten()[1:].to(torch.int64) // object_num
    return preds, targets


class dha_rnns():
    def __init__(self, input_size, hidden_size, num_layers, num_objects, net_name='rnn'):
        super(dha_rnns, self).__init__()
        self.num_objects = num_objects
        self.net_name = net_name
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        if self.net_name == 'rnn':
            self.model = RNN_Net(self.input_dim, self.hidden_dim, self.num_layers, self.num_objects)
        elif self.net_name == 'lstm':
            self.model = LSTM_Net(self.input_dim, self.hidden_dim, self.num_layers, self.num_objects)
        elif self.net_name == 'gru':
            self.model = GRU_Net(self.input_dim, self.hidden_dim, self.num_layers, self.num_objects)

        print('Current net is {}.'.format(self.net_name.upper()))

    def model_train(self, train_data, epoch: int, lr=0.002, device=torch.device('cpu')):
        print('Train begging.')
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        train_log = []
        for e in range(epoch):
            all_pred, all_target = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
            # for batch in tqdm(train_data):
            for index, (batch_x, batch_y) in enumerate(train_data):
                integrated_pred = self.model(batch_x.to(device))
                for student in range(batch_x.shape[0]):
                    pred, truth = process_pred(integrated_pred[student].to(device), batch_y[student].to(device),
                                               self.num_objects)
                    all_pred = torch.cat([all_pred, pred]).to(torch.float32)
                    all_target = torch.cat([all_target, truth]).to(torch.float32)

            loss = loss_function(all_pred, all_target)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = all_pred.cpu().detach()
            targets = all_target.cpu().detach()
            auc = roc_auc_score(targets, preds)
            acc = get_acc(torch.tensor(get_corrects(preds)), targets)
            train_log.append([epoch, loss.item(), auc, acc])
            # print("[Epoch {}] Loss: {:.3f}, AUC:{:.3f}, ACC:{:.3f}".format(e, loss, auc, acc))
        train_log = pd.DataFrame(data=train_log, columns=['Epochs', 'Train Loss', 'Train AUC', 'Train ACC'])
        train_log.to_excel('results/logs/rnns/{}_train_log.xlsx'.format(self.net_name.upper()), index=False)
        print('The train_log save succeed!')
        # return self.model

    def model_test(self, test_data, device=torch.device('cpu')):
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            y_pred, y_truth = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
            for index, (batch_x, batch_y) in tqdm(enumerate(test_data)):
                integrated_pred = self.model(batch_x.to(device))
                batch_size = batch_x.shape[0]
                for student in range(batch_size):
                    pred, truth = process_pred(integrated_pred[student], batch_y[student].to(device), self.num_objects)
                    y_pred = torch.cat([y_pred, pred])
                    y_truth = torch.cat([y_truth, truth])
            preds = y_pred.cpu().detach()
            targets = y_truth.cpu().detach()
            auc = roc_auc_score(targets, preds)
            acc = get_acc(torch.tensor(get_corrects(preds)), targets)
            test_log = [auc, acc]
            test_log = pd.DataFrame(data=[test_log], columns=['Test AUC', 'Test ACC'])
            test_log.to_excel('results/logs/rnns/{}_test_log.xlsx'.format(self.net_name.upper()), index=False)
            # print('The test_log save succeed!')
        return auc, acc

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
