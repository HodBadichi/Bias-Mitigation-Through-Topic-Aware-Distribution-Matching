import sys
sys.path.append('/home/mor.filo/nlp_project/')
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from DistributionMatching.text_utils import clean_abstracts, TextUtils
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def break_sentence_batch(samples):
    indexes = []
    all_sentences = []
    index = 0
    max_len = 0
    for sample in samples:
        sample_as_list = sample.split('<BREAK>')
        # sample is a list of sentences
        indexes.append((index, index + len(sample_as_list)))
        index += len(sample_as_list)
        all_sentences.extend(sample_as_list)
        if max_len < len(sample_as_list):
            max_len = len(sample_as_list)
    return indexes, all_sentences, max_len


class PubMedDataSetForBert(Dataset):
    def __init__(self, documents_dataframe):
        self.df = documents_dataframe
        self.tu = TextUtils()

    def __len__(self):
        # defines len of epoch
        return len(self.df)

    def __getitem__(self, index):
        return {'text': self.df.iloc[index]["broken_abstracts"]}

class PubMedModuleForBert(pl.LightningDataModule):
    def __init__(self):
        self.train = None
        self.val = None
        self.test = None
        self.documents_df = None

    def prepare_data(self):
        self.documents_df = pd.read_csv(rf'../../data/abstract_2005_2020_gender_and_topic.csv', encoding='utf8')
        train_df, testing_df = train_test_split(self.documents_df, test_size=0.7,random_state=42)
        test_df, val_df = train_test_split(testing_df, test_size=0.5, random_state=42)
        self.train_df = clean_abstracts(train_df)
        self.val_df = clean_abstracts(val_df)
        self.test_df = clean_abstracts(test_df)

    def setup(self, stage=None):
        self.train = PubMedDataSetForBert(self.train_df)
        self.val = PubMedDataSetForBert(self.val_df)
        self.test = PubMedDataSetForBert(self.test_df)

    def train_dataloader(self):
        # data set, batch size, shuffel, workers
        return DataLoader(self.train, shuffle=True, batch_size=10, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=10, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=10, num_workers=12)



class BertPretrain(pl.LightningModule):
    def __init__(self):
        super(BertPretrain, self).__init__()
        # bert_tokenizer : 'google/bert_uncased_L-2_H-128_A-2'
        self.tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        # bert_pretrained_path : 'google/bert_uncased_L-2_H-128_A-2'
        self.bert_model = BertForMaskedLM.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer)
        self.max_len = 70
        self.model_desc = 'bert_tiny_uncased'


    def forward(self, batch):
        # break the text into sentences
        indexes, all_sentences, max_len = break_sentence_batch(batch['text'])
        inputs = self.tokenizer(all_sentences, padding=True, truncation=True, max_length=self.max_len,
                                add_special_tokens=True, return_tensors="pt")
        # Collator output is a dict, and it will have 'input_ids' and 'labels'
        collated = self.data_collator(inputs['input_ids'].tolist())
        inputs['labels'] = collated['labels']
        inputs['input_ids'] = collated['input_ids']
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # At this point, inputs has the 'labels' key for LM and so the loss will not be None.
        x = self.bert_model(**inputs)
        return x.loss

    def step(self, batch: dict, name='train') -> dict:
        bert_loss = self.forward(batch)
        self.log(f'bert_{name}_loss', bert_loss, batch_size=10)
        return bert_loss

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        loss = self.step(batch, name='train')
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        path = os.path.join("/home/mor.filo/nlp_project/DistributionMatching/Models", rf"{self.model_desc}_abstract_2005_2020_gender_and_topic_{self.current_epoch}_")
        if self.current_epoch > 0 and not os.path.exists(path):
            self.bert_model.save_pretrained(path)
        loss = self.step(batch, name='val')
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        loss = self.step(batch, name='test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.bert_model.parameters(), lr=5e-5)
        return [optimizer]


if __name__ == '__main__':
    dm = PubMedModuleForBert()
    model = BertPretrain()
    trainer = pl.Trainer(gpus=1,
                         auto_select_gpus=True,
                         max_epochs=2,
                         log_every_n_steps=10,
                         accumulate_grad_batches=1,  # no accumulation
                         precision=16)
    trainer.fit(model, datamodule=dm)

