import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, BertForMaskedLM
from pytorch_lightning import Trainer
from DistributionMatching.utils import config
import numpy as np


class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()
        self.bert_tokenizer = AutoTokenizer.from_pretrained(config['models']['bert_tokenizer'])
        # TODO get bert_pretrained_path
        self.bert_model = BertForMaskedLM.from_pretrained(config['models']['bert_pretrained_path'])
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.classifier = nn.Linear(self.sentence_embedding_size * self.max_sentences, 1)
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self):
        # Forward is unneeded , GaN model will not infer in the future
        pass

    def step(self, batch: dict, optimizer_idx: int = None, name='train'):
        # batch is a dict of : bias doc, unbiased doc and the origin doc (the doc that was sampled by "get item"
        # and the matcher found its match)
        y_true = []
        texts = []
        for idx, data in enumerate(batch):
            coin_flip_result = np.random.binomial(1, 0.5)
            # Swap between 'bias' and 'unbiased' entries to confuse model
            if coin_flip_result:
                y_true.append((0, 1))  # (unbiased,biased)
                texts.append(data['unbiased'], data['biased'])
            else:
                y_true.append((1, 0))  # (biased,unbiased)
                texts.append(data['biased'], data['unbiased'])

        # We use the same bert model for all abstracts, regardless of year.
        inputs = self.bert_tokenizer.__call__(text_pair=texts,
                                              padding=True,
                                              truncation=True,
                                              max_length=self.max_len,
                                              add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.bert_model(**inputs, output_hidden_states=True).hidden_states[-1]
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def configure_optimizers(self):
        pass
