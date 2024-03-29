from torch.utils.data import Dataset
import sys
import os

if os.name != 'nt':
    sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))
"""
Data Set implementation.
A batch consists with cleaned title and abstract for bert (using CleanAbstracts)
"""


class FrozenBertDataSet(Dataset):
    def __init__(self, documents_dataframe):
        self.df = documents_dataframe

    def __len__(self):
        # defines len of epoch
        return len(self.df)

    def __getitem__(self, index):
        return {'text': self.df.iloc[index]["broken_abstracts"]}
