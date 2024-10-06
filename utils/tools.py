import numpy as np
import torch


def get_acc(outputs, targets):
    corrects_num = get_corrects_num(outputs, targets)
    return corrects_num / len(outputs)


def get_corrects_num(outputs, labels):
    outputs = get_corrects(outputs)
    corrects_num = torch.sum(torch.eq(torch.tensor(outputs), torch.tensor(labels))).item()

    return corrects_num


def get_corrects(outputs):
    modified_list = []
    for num in outputs:
        if num > 0.5:
            modified_list.append(1)
        else:
            modified_list.append(0)
    return modified_list


def get_pre_y(output):
    max_indices = np.argmax(np.array(output.cpu()), axis=1)
    max_indices = torch.tensor(max_indices, dtype=torch.float32, requires_grad=True).unsqueeze(-1)
    return max_indices


def get_loss_y(true_y):
    one_hot_y = F.one_hot(true_y.to(torch.int64).view(-1), num_classes=5)
    one_hot_y = torch.tensor(one_hot_y, dtype=torch.float32)
    return one_hot_y


def get_acc(all_pre_y, all_true_y):
    equal_elements = torch.eq(all_pre_y, all_true_y)
    num_equal = torch.sum(equal_elements.int()).item()
    acc = round(num_equal / all_pre_y.shape[0], 2)
    return acc


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def get_logs(model_log, path):
    model_log.to_excel(path, index=False)
    print('Results saved!')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation acc increased ({self.val_loss_min:.3f} --> {val_loss:.3f}).  Saving model ...')
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
