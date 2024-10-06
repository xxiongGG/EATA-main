from torch.utils.data import Dataset, DataLoader


class QCustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def q_get_dataloader(dataset, batch_size=4):
    dataset_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

    return dataset_loader


