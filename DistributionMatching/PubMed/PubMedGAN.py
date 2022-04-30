import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, BertForMaskedLM
from pytorch_lightning import Trainer
from DistributionMatching.utils import config
import numpy as np
import random
from PubMedModule import PubMedModule
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
import pytz
from DistributionMatching.text_utils import break_sentence_batch


class GAN(pl.LightningModule):
    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.hparams.bert_tokenizer)
        # TODO get bert_pretrained_path
        self.bert_model = BertForMaskedLM.from_pretrained(self.hparams.bert_pretrained_over_pubMed_path)
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.classifier = nn.Linear(self.sentence_embedding_size * self.max_sentences, 1)
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self):
        # Forward is unneeded , GaN model will not infer in the future
        pass

    def step(self, batch: dict, optimizer_idx: int = None, name='train'):
        """
        :param batch:{'origin_text':string,'biased':string,'unbiased':string,'origin':index}
        :param optimizer_idx: determines which step is it - discriminator or generator
        :param name:
        :return:
        """
        #   Discriminator Step
        if optimizer_idx == 0:
            mlm_loss = 0
            discriminator_loss = self.discriminator_step(batch)
            loss = discriminator_loss
            self.log(f'discriminator/{name}_loss', discriminator_loss)

        #   Generator Step
        if optimizer_idx == 1:
            discriminator_loss = self._discriminator_step(batch)
            loss, mlm_loss = self._generator_step(batch, discriminator_loss, name)

        return {'loss': loss,
                'mlm_loss': mlm_loss,
                'optimizer_idx': optimizer_idx}

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, optimizer_idx, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        # TODO Shunit: why do you preform both optimizers but only returns the loss of generator?
        outs = []
        for i in range(len(self.optimizers())):
            outs.append(self.step(batch, i, 'val'))
        return outs[1]  # generator

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        return self.validation_step(batch, batch_idx, optimizer_idx)

    def configure_optimizers(self):
        # Discriminator step paramteres -  classifier.
        grouped_parameters0 = [{'params': self.classifier.parameters()}]
        optimizer0 = torch.optim.Adam(grouped_parameters0, lr=self.hparams.learning_rate)
        # Generator step parameters - only 'the bert model.
        grouped_parameters1 = [{'params': self.bert_model.parameters()}]
        optimizer1 = torch.optim.Adam(grouped_parameters1, lr=self.hparams.learning_rate)
        return [optimizer0, optimizer1]

    """################# DISCRIMINATOR FUNCTIONS #####################"""

    def _discriminator_step(self, batch):
        clean_discriminator_batch = self._discriminator_clean_batch(batch)
        discriminator_y_true = [random.choice([0, 1]) for _ in clean_discriminator_batch]
        discriminator_predictions = self._discriminator_get_predictions(clean_discriminator_batch, discriminator_y_true)
        all_samples_losses = self.loss_func(discriminator_predictions, discriminator_y_true)
        discriminator_loss = all_samples_losses.mean(all_samples_losses)
        return discriminator_loss

    def _discriminator_clean_batch(self, batch):
        clean_batch = []  # batch where each document has a pair , no unmatched documents allowed
        for sample in batch:
            if sample['biased'] is None or sample['unbiased'] is None:
                continue
            clean_batch.append(sample)
        return clean_batch

    def _discriminator_get_batch(self, batch, shuffle_vector):
        result_batch = []
        for index, entry in enumerate(batch):
            biased_text = entry['biased']
            unbiased_text = entry['unbiased']
            if shuffle_vector[index] == 0:
                result_batch.append(biased_text)
                result_batch.append(unbiased_text)
            if shuffle_vector[index] == 1:
                result_batch.append(unbiased_text)
                result_batch.append(biased_text)
        return result_batch

    def _discriminator_get_cls_bert_outputs(self, bert_inputs):
        all_outputs = self.bert_model(**bert_inputs, output_hidden_states=True).hidden_states[-1]
        cls_outputs = all_outputs[:, 0]
        return cls_outputs

    def _discriminator_bert_embeddings_to_predictions(self, bert_cls_outputs, begin_end_indexes):
        sample_embedding = []

        def fix_sentences_size(sent_embeddings):
            """
            Fix embedding size in case it requires padding or truncating

            :param sent_embeddings:
            :return:
            """
            if len(sent_embeddings[start_index_first_document, end_index_first_document]) > self.max_sentences:  # 
                # Too many sentences 
                truncated_embeddings = sent_embeddings[:self.max_sentences]
                return torch.flatten(truncated_embeddings)
            else:
                padding = torch.zeros(self.max_sentences - len(sent_embeddings), self.sentence_embedding_size,
                                      device=self.device)
                return torch.flatten(torch.cat([sent_embeddings, padding], dim=0))

        for i in range(0, len(begin_end_indexes), 2):
            start_index_first_document, end_index_first_document = begin_end_indexes[i]
            start_index_second_document, end_index_second_document = begin_end_indexes[i + 1]
            first_document_embeddings = fix_sentences_size(
                bert_cls_outputs[start_index_first_document, end_index_first_document])
            second_document_embeddings = fix_sentences_size(
                bert_cls_outputs[start_index_second_document, end_index_second_document])
            curr_concat_embeddings = torch.cat(first_document_embeddings, second_document_embeddings)
            sample_embedding.append(curr_concat_embeddings)

        aggregated = torch.stack(sample_embedding)
        y_predictions = self.classifier(aggregated).squeeze(1)
        return y_predictions

    def _discriminator_get_predictions(self, batch, shuffle_vector):
        """

        :param batch: a batch of PubMedGan in the shape of {'origin':int,'biased':string,'unbiased':string}
        might include `None` values in the `biased` and `unbiased` entry in case the origin document has no match.

        :return: This function wil return the classifier predictions over bertmodel output embeddings and the
        "ground truth" a shuffle_vector: list in the shape of [0,...,1,0...] which can be interpreted like that:
        1 - the couple of matching docs was [biased,unbiased]
        0 - the couple of matching docs was [unbiased,biased]
        """

        discriminator_batch = self._discriminator_get_batch(batch, shuffle_vector)

        begin_end_indexes, documents_sentences, max_len = break_sentence_batch(discriminator_batch)

        bert_inputs = self._get_bert_inputs(documents_sentences)

        # `bert_all_outputs` is not being used.
        bert_cls_outputs = self._discriminator_get_cls_bert_outputs(bert_inputs)
        return self._discriminator_bert_embeddings_to_predictions(bert_cls_outputs, begin_end_indexes)

    """################# GENERATOR FUNCTIONS #####################"""
    def _generator_step(self, batch, discriminator_loss, name):
        generator_batch = self._get_generator_batch(batch)
        begin_end_indexes, documents_sentences, max_len = break_sentence_batch(generator_batch)
        bert_inputs = self._get_bert_inputs(documents_sentences)
        mlm_loss = self._get_generator_mlm_loss(bert_inputs)
        loss = self.hparams.mlm_factor * mlm_loss + self.hparams.discriminator_factor * discriminator_loss
        self.log(f'generator/{name}_loss', loss)
        self.log(f'generator/{name}_mlm_loss', mlm_loss)
        self.log(f'generator/{name}_discriminator_loss', discriminator_loss)
        return loss, mlm_loss

    def _generator_get_mlm_loss(self, inputs):
        """returns MLM loss"""
        collated_inputs = self.data_collator(inputs['input_ids'].tolist())
        collated_inputs = {k: v.to(self.device) for k, v in collated_inputs.items()}

        inputs['input_ids'] = collated_inputs['input_ids']
        inputs['labels'] = collated_inputs['labels']
        loss = self.bert_model(**inputs).loss
        return loss

    def _generator_get_batch(self, batch):
        result_batch = []
        for index, entry in enumerate(batch):
            result_batch.append(entry['origin_text'])
        return result_batch

    """################# UTILS FUNCTIONS #####################"""

    def _get_bert_inputs(self, documents_sentences):
        inputs = self.bert_tokenizer.batch_encode_plus(documents_sentences, padding=True, truncation=True,
                                                       max_length=self.max_len,
                                                       add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs


hparams = {'learning_rate': 5e5,
           'discriminator_factor': 0.5,
           'mlm_factor': 0.5,
           'gpus': 1,
           'max_epochs': 40,
           'gender_and_topic_path': '../../data/abstract_2005_2020_gender_and_topic.csv',
           'batch_size': 64,
           'test_size': 0.7,
           'bert_pretrained_over_pubMed_path': '',
           'bert_tokenizer': 'google/bert_uncased_L-2_H-128_A-2'
           }

if __name__ == '__main__':
    dm = PubMedModule(hparams)
    model = GAN(hparams)
    logger = WandbLogger(name=f'GAN_over_topic_and_gender_70_15_15',
                         version=datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f'),
                         project='GAN_test',
                         config={'lr': 5e-5, 'batch_size': 16})
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         log_every_n_steps=20,
                         accumulate_grad_batches=1,
                         num_sanity_val_steps=0,
                         # gradient_clip_val=0.3
                         )
    trainer.fit(model, datamodule=dm)
