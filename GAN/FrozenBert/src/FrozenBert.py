import os
import torch

from transformers import AutoTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
import pytorch_lightning as pl

from GAN.Utils.TextUtils import BreakSentenceBatch

"""
FrozenBert implementation , the created bert model is used as a frozen model for GAN and also as
the starting encoder for the GAN generator.
The training of the model is on MLM task with GANPubMed data.
"""


class FrozenBert(pl.LightningModule):
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        super(FrozenBert, self).__init__()
        # bert_tokenizer : 'google/bert_uncased_L-2_H-128_A-2'
        self.tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        # bert_pretrained_path : 'google/bert_uncased_L-2_H-128_A-2'
        self.bert_model = BertForMaskedLM.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer)
        self.max_len = 70
        self.model_desc = 'bert_tiny_uncased'
        self.count = 0
        os.makedirs('models', exist_ok=True)

    def forward(self, batch):
        # break the text into sentences
        indexes, all_sentences, max_len = BreakSentenceBatch(batch['text'])
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

    def step(self, batch: dict, name='train_dataset') -> dict:
        bert_loss = self.forward(batch)
        self.log(f'bert_{name}_loss', bert_loss, batch_size=10)
        return bert_loss

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        self.count = self.count + 1
        loss = self.step(batch, name='train_dataset')
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        path = os.path.join('models', rf"{self.model_desc}_{self.current_epoch}_")
        if self.current_epoch > 0 and not os.path.exists(path):
            self.bert_model.save_pretrained(path)
        loss = self.step(batch, name='val_dataset')
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        loss = self.step(batch, name='test_dataset')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.bert_model.parameters(), lr=5e-5)
        return [optimizer]
