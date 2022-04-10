from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class PubMedDataSet(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        # defines len of epoch
        pass

    def __getitem__(self, item):
        # returns dict of 2 docs - bias\unbiased
        # more bias: title+abs, less bias: title+abs
        pass


class PubMedModule(pl.LightiningDataModule):
    def __init__(self):
        pass

    def prepare_data(self):
        # run before setup, 1 gpu
        pass

    def setup(self):
        # runs on all gpus
        # data set instanses (val, train, test)
        pass

    def train_dataloader(self):
        # data set, batch size, shuffel, workers
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass

