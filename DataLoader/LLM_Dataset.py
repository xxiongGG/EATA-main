from torch.utils.data import DataLoader, Dataset


class LLMCustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        if self.target is None:
            return self.data[index]

        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def llm_get_dataloader(dataset, batch_size=4):
    dataset_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

    return dataset_loader
