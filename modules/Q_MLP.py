import numpy as np
import torch
import torch.nn as nn
from torch import optim

from DataLoader.Q_Dataset import q_get_dataloader
from layers.focal_loss_v1 import FocalLoss
from utils.tools import get_acc, get_pre_y, get_loss_y


class MLP_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device='cpu', dropout_rate=0.5):
        super(MLP_Classifier, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.num_classes = num_classes
        self.device = device

        self.fc_1 = nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
        self.fc_2 = nn.Linear(self.hidden_dim, int(self.hidden_dim * 0.5))
        self.fc_3 = nn.Linear(int(self.hidden_dim * 0.5), self.num_classes)
        self.relu = nn.ReLU().to(self.device)
        self.dropout = nn.Dropout(dropout_rate, inplace=False)
        self.softmax = nn.Softmax()
        self.layer_norm = nn.LayerNorm(self.input_dim)

    def forward(self, x):
        x_normed = self.layer_norm(x)
        h_1 = self.relu(self.fc_1(x_normed))
        h_1 = self.dropout(h_1)
        h_2 = self.fc_2(h_1)
        h_2 = self.dropout(h_2)
        output = self.fc_3(h_2)
        output = self.softmax(output)
        # h_2 is our question embedding.
        return output, h_2


def model_train(model, train_dataloader, criterion, optimizer, device=torch.device('cpu'), epochs=200,
                loss_name='focal'):
    print('Beginning train. Current device is: {}.'.format(device))
    for epoch in range(epochs):
        all_pre_y = torch.tensor([]).to(device)
        all_true_y = torch.tensor([]).to(device)
        loss_all = []
        for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
            output, _ = model(batch_x)

            if loss_name == 'focal':
                loss_y = batch_y.to(torch.int64).view(-1)
            else:
                loss_y = get_loss_y(batch_y.detach()).to(device)

            loss = criterion(output, loss_y)
            loss_all.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pre_y = get_pre_y(output.detach()).to(device)
            all_pre_y = torch.cat([all_pre_y, pre_y.view(-1)])
            all_true_y = torch.cat([all_true_y, batch_y.view(-1)])
        loss_mean = sum(loss_all) / len(loss_all)
        acc = get_acc(all_pre_y, all_true_y)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_mean:.4f}, Acc: {acc:.4f}')


def model_test(model, test_dataloader, device=torch.device('cpu')):
    print('Beginning test. Current device is: {}.'.format(device))
    model.eval()
    with torch.no_grad():
        all_pre_y = torch.tensor([]).to(device)
        all_true_y = torch.tensor([]).to(device)
        for batch_idx, (batch_x, batch_y) in enumerate(test_dataloader):
            output, _ = model(batch_x)
            pre_y = get_pre_y(output.detach()).to(device)
            all_pre_y = torch.cat([all_pre_y, pre_y.view(-1)])
            all_true_y = torch.cat([all_true_y, batch_y.view(-1)])
    acc = get_acc(all_pre_y, all_true_y)
    print(f'Test Acc: {acc:.4f}')


def model_q_embedding(model, all_dataloader, device=torch.device('cpu')):
    print('Beginning get questions embedding matrix.')
    model.eval()
    with torch.no_grad():
        q_embedding_matrix = torch.tensor([]).to(device)
        for batch_idx, (batch_x, batch_y) in enumerate(all_dataloader):
            _, q_embedding = model(batch_x)
            q_embedding_matrix = torch.cat([q_embedding_matrix, q_embedding])
    q_embedding_matrix = q_embedding_matrix.cpu().numpy()
    print('Questions embedding matrix shape is: {}.'.format(q_embedding_matrix.shape))
    return q_embedding_matrix


if __name__ == '__main__':
    input_size = 384
    hidden_size = 128
    num_classes = 5
    learning_rate = 0.0001
    epochs = 200
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP_Classifier(input_size, hidden_size, num_classes).to(device)
    # [2, 2, 1, 2, 3] come from data distribution.
    criterion = FocalLoss(class_num=5, alpha=torch.tensor([2, 2, 1, 2, 3], device=device), gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_q_dataset = torch.load(r'../data/q/train_q_dataset.pth',
                                 map_location=device)

    train_dataloader = q_get_dataloader(train_q_dataset)

    model_train(model,
                train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=epochs,
                loss_name='focal')

    torch.save(model, r'../results/Q_HOTS_MLPClassifier.pth')

    test_q_dataset = torch.load(r'../data/q/test_q_dataset.pth',
                                map_location=device)

    test_dataloader = q_get_dataloader(test_q_dataset)

    model = torch.load(r'../results/Q_HOTS_MLPClassifier.pth')
    model_test(model, test_dataloader, device)

    all_q_dataset = torch.load(r'../data/q/all_q_dataset.pth',
                               map_location=device)

    all_dataloader = q_get_dataloader(all_q_dataset)
    q_embedding = model_q_embedding(model, all_dataloader, device)
    np.save("../data/q_embedding", q_embedding)
