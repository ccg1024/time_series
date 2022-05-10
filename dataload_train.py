"""
training model by patched data object.
"""
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class CustomDataset(Dataset):
    """
    A simple framework for dataset object. the function parameter maybe need to add.
    and the content of function need to be finished.
    """
    def __ini__(self, transform=None, target_transform=None):
        """
        :param transform: a modify function or a Lambda expression. To change the data to final version.
        :param target_transform: just like `transform`, modify the target data.
        """
        # make some prepare for raw data,
        # example: reshape the data.
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # return the total len of training data.
        pass

    def __getitem__(self):
        # return one sample of training data.
        # the return form: x, y.

        if self.transform:
            pass
        if self.target_transform:
            pass
        pass


# create dataset obj
# the DataLoader obj is a iterator. return a batched X, y.
training_data = CustomDataset()
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
