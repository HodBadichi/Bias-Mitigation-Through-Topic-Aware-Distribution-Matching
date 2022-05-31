import sys

sys.path.append('/home/mor.filo/nlp_project/')
import torch
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from GAN.PubMed.text_utils import clean_abstracts, TextUtils, break_sentence_batch
import os
import pytorch_lightning as pl

os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
BertPretrain implementation , the created bert model is used a freezed model for GAN and also as
the starting encoder for the GAN generator.
The training of the model is on MLM task with PubMed data.
"""

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
        self.count = 0

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
        self.count = self.count + 1
        loss = self.step(batch, name='train')
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        path = os.path.join("/home/mor.filo/nlp_project/DistributionMatching/Models",
                            rf"{self.model_desc}_abstract_2005_2020_gender_and_topic_{self.current_epoch}_")
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