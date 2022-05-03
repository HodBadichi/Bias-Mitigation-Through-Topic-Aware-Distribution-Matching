import sys

sys.path.append('/home/mor.filo/nlp_project/')

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
import random
from DistributionMatching.text_utils import break_sentence_batch
from sklearn.metrics import roc_auc_score, accuracy_score
import wandb
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pytz
import pprint


class PubMedGAN(pl.LightningModule):
    def __init__(self, hparams):
        super(PubMedGAN, self).__init__()
        self.hparams.update(hparams)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.hparams['bert_tokenizer'])
        self.data_collator = DataCollatorForLanguageModeling(self.bert_tokenizer)
        self.max_length_bert_input = self.hparams['max_length_bert_input']
        self.max_sentences_per_abstract = self.hparams['max_sentences_per_abstract']
        self.bert_model = BertForMaskedLM.from_pretrained(self.hparams['bert_pretrained_over_pubMed_path'])
        # self.frozen_bert_model = BertForMaskedLM.from_pretrained(self.hparams['bert_pretrained_over_pubMed_path'])
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        # The linear layer if from 2 concat abstract (1 is bias and 1 unbiased) to binary label:
        # 1 - the couple of matching docs was [biased,unbiased]
        # 0 - the couple of matching docs was [unbiased,biased]
        self.classifier = nn.Linear(self.sentence_embedding_size * self.max_sentences_per_abstract * 2, 1)
        # todo : why reduction='none'
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self):
        # Forward is unneeded , GaN model will not infer in the future
        pass

    def step(self, batch: dict, optimizer_idx: int = None, name='train'):
        """
        :param batch:{'origin_text':list[string],'biased':list[string],'unbiased':list[string]}
        :param optimizer_idx: determines which step is it - discriminator or generator
        :param name:
        :return:
        """
        #   Discriminator Step
        step_ret_dict = {}
        # {'loss': , 'losses': , 'mlm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        if optimizer_idx == 0:
            step_ret_dict = self._discriminator_step(batch, name)
            print('_discriminator_step_ret_dict')
            pprint.pprint(step_ret_dict)
        #   Generator Step
        if optimizer_idx == 1:
            step_ret_dict = self._discriminator_step(batch, name)
            step_ret_dict = self._generator_step(batch, step_ret_dict, name)
            print('_generator_step_ret_dict')
            pprint.pprint(step_ret_dict)
        return step_ret_dict

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, optimizer_idx, 'train')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        outs = []
        for i in range(len(self.optimizers())):
            outs.append(self.step(batch, i, 'val'))
        # return value doesn't meter, its only pl requirement, the logging is with wandb
        return outs[1]  # generator

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        return self.validation_step(batch, batch_idx, optimizer_idx)

    def test_epoch_end(self, outputs) -> None:
        self.log('test/val_mlm_loss_avg', torch.Tensor([output['mlm_loss'] for output in outputs]).mean())
        if self.hparams['max_epochs'] > 0:
            dir_path = os.path.join(self.hparams['SAVE_PATH'],
                                    f"bert_GAN_model_{datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f')}")
            self.bert_model.save_pretrained(os.path.join(dir_path, f'epoch{self.hparams["max_epochs"] - 1}'))
            if not os.path.exists(f'{dir_path}/config'):
                np.save(f'{dir_path}/config', self.hparams)

    def training_epoch_end(self, outputs):
        self.on_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.on_end(outputs, 'val')

    def on_end(self, outputs, name):
        # outputs is a list (len=number of batches) of dicts (as returned from the step methods).
        if name == 'train':
            outputs = outputs[0]  # TODO: WHY? only generator outputs. TODO: really?
        losses = torch.cat([output['losses'] for output in outputs])
        y_true = torch.cat([output['y_true'] for output in outputs])
        y_proba = torch.cat([output['y_proba'] for output in outputs])
        y_score = torch.cat([output['y_score'] for output in outputs])

        self.log(f'debug/{name}_loss_histogram', wandb.Histogram(losses.cpu()))
        self.log(f'debug/{name}_probability_histogram', wandb.Histogram(y_proba.cpu()))
        self.log(f'debug/{name}_score_histogram', wandb.Histogram(y_score.cpu()))
        self.log(f'debug/{name}_loss', losses.mean())
        self.log(f'debug/{name}_accuracy', (1. * ((1. * (y_proba >= 0.5)) == y_true)).mean())
        self.log(f'debug/{name}_1_accuracy', (1. * (y_proba[y_true == 1] >= 0.5)).mean())
        self.log(f'debug/{name}_0_accuracy', (1. * (y_proba[y_true == 0] < 0.5)).mean())

        # if name == 'val':
        #     texts = np.concatenate([output['text'] for output in outputs])
        #
        #     df = pd.DataFrame({'text': texts, 'y_true': y_true, 'y_score': y_score, 'y_proba': y_proba, 'loss': losses})
        #     df = df[(df['loss'] <= df['loss'].quantile(0.05)) | (df['loss'] >= df['loss'].quantile(0.95))]
        #     self.log(f'debug/{name}_table', wandb.Table(dataframe=df))

    def configure_optimizers(self):
        # Discriminator step paramteres -  classifier.
        grouped_parameters_discriminator = [{'params': self.classifier.parameters()}]
        optimizer_discriminator = torch.optim.Adam(grouped_parameters_discriminator, lr=self.hparams['learning_rate'])
        # Generator step parameters - only 'the bert model.
        grouped_parameters_generator = [{'params': self.bert_model.parameters()}]
        optimizer_generator = torch.optim.Adam(grouped_parameters_generator, lr=self.hparams['learning_rate'])
        return [optimizer_discriminator, optimizer_generator]

    """################# DISCRIMINATOR FUNCTIONS #####################"""

    def _y_pred_to_probabilities(self, y_pred):
        return torch.sigmoid(y_pred)

    def _discriminator_step(self, batch, name):
        """
        :param batch:{'origin_text':string,'biased':string,'unbiased':string}
        :param name:
        :return: result dictionary
        since not all batch items represent a couple of docs to discriminator (some didn't get match with noahArc matcher)
        we clean (leave) the relevant docs in the batch, shuffle them, get prediction and return loss
        """
        result_dictionary = {}
        # {'loss': , 'losses': , 'mlm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        print(1)
        result_dictionary['mlm_loss'] = 0
        result_dictionary['optimizer_idx'] = 0
        clean_discriminator_batch = self._discriminator_clean_batch(batch)
        discriminator_y_true = torch.as_tensor([float(random.choice([0, 1])) for _ in clean_discriminator_batch])
        print(2)
        result_dictionary['y_true'] = discriminator_y_true
        # discriminator_y_true created in order to shuffle the bias/unbiased order
        discriminator_predictions = self._discriminator_get_predictions(clean_discriminator_batch, discriminator_y_true)
        all_samples_losses = self.loss_func(discriminator_predictions, discriminator_y_true.to(self.device))
        print(3)
        discriminator_loss = all_samples_losses.mean()
        result_dictionary['loss'] = discriminator_loss
        result_dictionary['losses'] = all_samples_losses
        self.log(f'discriminator/{name}_loss', discriminator_loss)
        y_proba = self._y_pred_to_probabilities(discriminator_predictions).cpu().detach()
        result_dictionary['y_proba'] = y_proba
        print(4)
        result_dictionary['y_score'] = discriminator_y_true.cpu().detach() * y_proba + (
                    1 - discriminator_y_true.cpu().detach()) * (1 - y_proba)
        if not all(discriminator_y_true) and any(discriminator_y_true):
            # Calc auc only if batch has more than one class.
            self.log(f'discriminator/{name}_auc', roc_auc_score(discriminator_y_true.cpu().detach(), y_proba))
        self.log(f'discriminator/{name}_accuracy', accuracy_score(discriminator_y_true.cpu().detach(), y_proba.round()))
        return result_dictionary

    def _discriminator_clean_batch(self, batch):
        clean_batch = []  # batch where each document has a pair , no unmatched documents allowed
        for sample in batch:
            if sample['biased'] == "" or sample['unbiased'] == "":
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
            if len(sent_embeddings) > self.max_sentences_per_abstract:  #
                # Too many sentences 
                truncated_embeddings = sent_embeddings[:self.max_sentences_per_abstract]
                return torch.flatten(truncated_embeddings)
            else:
                padding = torch.zeros(self.max_sentences_per_abstract - len(sent_embeddings),
                                      self.sentence_embedding_size,
                                      device=self.device)
                return torch.flatten(torch.cat([sent_embeddings, padding], dim=0))

        for i in range(0, len(begin_end_indexes), 2):
            start_index_first_document, end_index_first_document = begin_end_indexes[i]
            start_index_second_document, end_index_second_document = begin_end_indexes[i + 1]
            first_document_embeddings = fix_sentences_size(
                bert_cls_outputs[start_index_first_document: end_index_first_document])
            second_document_embeddings = fix_sentences_size(
                bert_cls_outputs[start_index_second_document: end_index_second_document])
            curr_concat_embeddings = torch.cat((first_document_embeddings, second_document_embeddings))
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

        bert_cls_outputs = self._discriminator_get_cls_bert_outputs(bert_inputs)
        return self._discriminator_bert_embeddings_to_predictions(bert_cls_outputs, begin_end_indexes)

    """################# GENERATOR FUNCTIONS #####################"""

    def _generator_step(self, batch, discriminator_step_ret_dict, name):
        step_ret_dict = discriminator_step_ret_dict
        step_ret_dict['optimizer_idx'] = 1
        discriminator_loss = discriminator_step_ret_dict['loss']
        print(5)
        # {'loss': , 'losses': , 'mlm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        generator_batch = self._generator_get_batch(batch)
        begin_end_indexes, documents_sentences, max_len = break_sentence_batch(generator_batch)
        bert_inputs = self._get_bert_inputs(documents_sentences)
        mlm_loss = self._generator_get_mlm_loss(bert_inputs)
        print(6)
        step_ret_dict['mlm_loss'] = mlm_loss
        total_loss = self.hparams['mlm_factor'] * mlm_loss - self.hparams['discriminator_factor'] * discriminator_loss
        step_ret_dict['loss'] = total_loss
        print(7)
        # TODO diff from frozen and tune the factors
        self.log(f'generator/{name}_loss', total_loss)
        self.log(f'generator/{name}_mlm_loss', mlm_loss)
        self.log(f'generator/{name}_discriminator_loss', discriminator_loss)
        return step_ret_dict

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

    def _convert_to_list_of_dicts(self, batch):
        batch_df = pd.DataFrame.from_dict(batch)
        batch = batch_df.T.to_dict().values()
        return batch
