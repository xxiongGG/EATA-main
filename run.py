import argparse
from os.path import join as opj

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataLoader.LLM_Dataset import llm_get_dataloader
from modules.DHA_LLM import dha_llm
from utils.tools import EarlyStopping, get_acc, get_corrects, get_logs
from layers.focal_loss_v1 import FocalLoss
from layers.focal_loss_v2 import FocalLossV1
import pandas as pd

parser = argparse.ArgumentParser(description='GPT4HA')

parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--train_epochs', type=int, default=30)
# 64
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--object_num', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--gpt_layers', type=int, default=1)
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=20)
parser.add_argument('--model_path', type=str, default=r'E:\0-HGroup_E\2023-xiaoxiong\PLM_Model\gpt2')


def get_data_loader(data_path, batch_size, shuffle=False):
    data = torch.FloatTensor(np.load(data_path))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def process_pred(raw_pred, raw_true, object_num: int) -> tuple:
    objects = raw_true.flatten().to(torch.int64) % object_num
    length = objects.shape[0]
    preds = raw_pred[:length]
    preds = preds.gather(1, objects.view(-1, 1)).flatten()[:-1]
    targets = raw_true.flatten()[1:].to(torch.int64) // object_num
    return preds, targets


def get_FocalLoss_pred(preds):
    preds = get_corrects(preds)
    num_classes = np.max(preds) + 1
    fl_preds = torch.tensor(np.eye(num_classes)[preds], dtype=torch.float32, requires_grad=True)
    return fl_preds


def model_train(train_data, vail_data, model, device, criterion, optimizer, log_path):
    early_stopping = EarlyStopping(patience=3, verbose=True)
    train_log = []
    for epoch in range(args.train_epochs):
        train_loss = []
        all_preds, all_targets = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        for index, (batch_x, batch_y) in tqdm(enumerate(train_data), "Epoch %s" % epoch):
            batch_preds, batch_targets = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
            model_pred = model(batch_x.to(device))
            for student in range(batch_x.shape[0]):
                pred, target = process_pred(model_pred[student], batch_y[student].to(device), 5)
                batch_preds = torch.cat([batch_preds, pred]).to(torch.float32)
                batch_targets = torch.cat([batch_targets, target]).to(torch.float32)
            batch_loss = criterion(batch_preds, batch_targets)
            train_loss.append(batch_loss.item())
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            all_preds = torch.cat([all_preds, batch_preds])
            all_targets = torch.cat([all_targets, batch_targets])
        all_targets = all_targets.to(torch.int32).detach().cpu()
        all_preds = all_preds.detach().cpu()
        auc = roc_auc_score(all_targets, all_preds)
        acc = get_acc(all_targets, torch.tensor(get_corrects(all_preds)))
        print("[Epoch {}] Loss: {:.3f} Auc: {:.3f} Acc: {:.3f}.".format(epoch,
                                                                        round(sum(train_loss) / len(train_data), 2),
                                                                        auc,
                                                                        acc))
        vali_loss, vail_acc, vali_auc = model_vali(model, vail_data, criterion, device)
        train_log.append([epoch, round(sum(train_loss) / len(train_data), 2), auc, acc, vali_loss, vali_auc, vail_acc])
        early_stopping(vail_acc, model, 'results/')
    train_log = pd.DataFrame(data=train_log,
                             columns=['Epochs', 'Train Loss', 'Train AUC', 'Train ACC', 'vali Loss', 'vail_auc',
                                      'vali_acc'])
    get_logs(train_log, opj(log_path, 'LLM_train_log.xlsx'))

    return model


def model_vali(model, vali_data, criterion, device):
    model.in_layer.eval()
    model.out_layer.eval()
    model.sigmoid.eval()
    model.dropout.eval()
    model.pred_layer.eval()

    all_preds, all_targets = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
    with torch.no_grad():
        for index, (batch_x, batch_y) in enumerate(vali_data):
            integrated_pred = model(batch_x.to(device))
            batch_size = batch_x.shape[0]
            for student in range(batch_size):
                pred, truth = process_pred(integrated_pred[student], batch_y[student].to(device), object_num=5)
                all_preds = torch.cat([all_preds, pred])
                all_targets = torch.cat([all_targets.to(device), truth.float().to(device)])
    vali_loss = criterion(all_preds, all_targets).item()
    vali_auc = roc_auc_score(all_targets.detach().cpu().numpy(), all_preds.detach().cpu().numpy())
    vali_acc = get_acc(all_targets, torch.tensor(get_corrects(all_preds)).to(device))

    model.in_layer.train()
    model.out_layer.train()
    model.sigmoid.train()
    model.dropout.train()
    model.pred_layer.train()

    return vali_loss, vali_acc, vali_auc


def model_test(test_data, model, device, log_path):
    model.eval()
    y_pred, y_truth = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
    with torch.no_grad():
        for index, (batch_x, batch_y) in tqdm(enumerate(test_data)):
            integrated_pred = model(batch_x.to(device))
            batch_size = batch_x.shape[0]
            for student in range(batch_size):
                pred, truth = process_pred(integrated_pred[student], batch_y[student].to(device), 5)
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth.to(device), truth.to(device)])
        auc = roc_auc_score(y_truth.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        acc = get_acc(y_truth.detach().cpu(), torch.tensor(get_corrects(y_pred)).detach().cpu())
        test_log = pd.DataFrame([[auc, acc]], columns=['Test AUC', 'Test ACC'])
    get_logs(test_log, opj(log_path, 'LLM_test_log.xlsx'))

    print("[Test] Auc: {:.3f} Acc: {:.3f}.".format(auc, acc))


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_step = 100
    llm_data_path = opj(r'data/llm', 'seq_{}'.format(max_step))
    log_path = opj(r'results/logs/llm', 'seq_{}'.format(max_step))
    input_size = 128

    train_dataset = torch.load(opj(llm_data_path, 'llm_train_dataset.pth'),
                               map_location=device)
    test_dataset = torch.load(opj(llm_data_path, 'llm_test_dataset.pth'),
                              map_location=device)
    vail_dataset = torch.load(opj(llm_data_path, 'llm_vail_dataset.pth'),
                              map_location=device)
    train_loader = llm_get_dataloader(train_dataset, batch_size=args.batch_size)
    test_loader = llm_get_dataloader(test_dataset, batch_size=1)
    vail_dataset = llm_get_dataloader(vail_dataset, batch_size=1)

    print('Train data len is: {}, test data len is: {}.'.format(len(train_loader), len(test_loader)))

    model = dha_llm(input_size, args, device)
    print(model)
    model.to(device)
    params = model.parameters()
    opt = torch.optim.Adam(params, lr=args.learning_rate)
    criterion = nn.BCELoss()
    model_train(train_data=train_loader,
                vail_data=vail_dataset,
                model=model,
                device=device,
                criterion=criterion,
                optimizer=opt,
                log_path=log_path)

    model.load_state_dict(torch.load('results/checkpoint.pth'))
    model_test(test_data=test_loader,
               model=model,
               device=device,
               log_path=log_path)

    # model_test(test_loader, model, device, num_q)
