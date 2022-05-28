import sys
import random
import torch
import os
if os.name != 'nt':
    sys.path.append('/home/mor.filo/nlp_project/')

from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import pytorch_lightning as pl
from torch import nn
from transformers import AutoTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
from DistributionMatching.text_utils import break_sentence_batch


class PubMedDiscriminator(pl.LightningModule):
    def __init__(self, hparams):
        super(PubMedDiscriminator, self).__init__()
        self.hparams.update(hparams)
        self.SentenceTransformerModel = SentenceTransformer('all-MiniLM-L6-v2')
        # The linear layer if from 2 concat abstract (1 is bias and 1 unbiased) to binary label:
        # 1 - the couple of matching docs was [biased,unbiased]
        # 0 - the couple of matching docs was [unbiased,biased]

        self.input_dropout = nn.Dropout(p=self.hparams['dropout_rate'])
        layers = []
        hidden_sizes = [self.sentence_embedding_size * self.max_sentences_per_abstract * 2] + self.hparams['hidden_sizes']
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [nn.Linear(hidden_sizes[i],
                           hidden_sizes[i + 1]),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Dropout(self.hparams['dropout_rate'])])

        self.layers = nn.Sequential(*layers)  # per il flatten


        # self.classifier = nn.Linear(self.sentence_embedding_size * self.max_sentences_per_abstract * 2, 1)
        # # todo : why reduction='none'
        # self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.empty_batch_count = 0

    def forward(self):
        # Forward is unneeded , GaN model will not infer in the future
        pass

    def step(self, batch: dict, name='train'):
        """
        :param batch:{'origin_text':list[string],'biased':list[string],'unbiased':list[string]}
        :param optimizer_idx: determines which step is it - discriminator or generator
        :param name:
        :return:
        """
        print("Stepped in")
        batch = self._convert_to_list_of_dicts(batch)
        #   Discriminator Step
        # {'loss': , 'losses': , 'mlm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        step_ret_dict = self._discriminator_step(batch, name)
        return step_ret_dict

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, 'val')

    def test_step(self, batch: dict, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # Discriminator step paramteres -  classifier.
        grouped_parameters_discriminator = [{'params': self.classifier.parameters()}]
        optimizer_discriminator = torch.optim.Adam(grouped_parameters_discriminator, lr=self.hparams['learning_rate'])
        return [optimizer_discriminator]

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
        assert (len(batch) > 0)
        clean_discriminator_batch = self._discriminator_clean_batch(batch)
        if len(clean_discriminator_batch) == 0:
            self.empty_batch_count += 1
            return None
        assert (len(clean_discriminator_batch) > 0)
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
        self.log(f'discriminator/{name}_loss', discriminator_loss, batch_size=self.hparams['batch_size'])
        y_proba = self._y_pred_to_probabilities(discriminator_predictions).cpu().detach()
        result_dictionary['y_proba'] = y_proba
        print(4)
        result_dictionary['y_score'] = discriminator_y_true.cpu().detach() * y_proba + (
                1 - discriminator_y_true.cpu().detach()) * (1 - y_proba)
        if not all(discriminator_y_true) and any(discriminator_y_true):
            # Calc auc only if batch has more than one class.
            self.log(f'discriminator/{name}_auc', roc_auc_score(discriminator_y_true.cpu().detach(), y_proba),
                     batch_size=self.hparams['batch_size'])
        self.log(f'discriminator/{name}_accuracy', accuracy_score(discriminator_y_true.cpu().detach(), y_proba.round()),
                 batch_size=self.hparams['batch_size'])
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


    def _discriminator_bert_embeddings_to_predictions(self, sentence_embeddings):
        sample_embedding = []

        for i in range(0, len(sentence_embeddings), 2):
            first_document_embeddings = sentence_embeddings[i]
            second_document_embeddings = sentence_embeddings[i+1]
            curr_concat_embeddings = torch.cat((first_document_embeddings, second_document_embeddings, abs(first_document_embeddings-second_document_embeddings)))
            sample_embedding.append(curr_concat_embeddings)

        aggregated = torch.stack(sample_embedding)
        y_predictions = self.classifier(aggregated).squeeze(1)
        return y_predictions

    def _discriminator_get_predictions(self, batch, shuffle_vector):
        """
        :param batch: a batch of PubMedGan in the shape of {'origin':int,'biased':string,'unbiased':string}
        might include `None` values in the `biased` and `unbiased` entry in case the origin document has no match.

        :return: This function wil return the classifier predictions over bertmodel output embeddings
        shuffle_vector: list in the shape of [0,...,1,0...] which can be interpreted like that:
        1 - the couple of matching docs was [biased,unbiased]
        0 - the couple of matching docs was [unbiased,biased]
        """

        discriminator_batch = self._discriminator_get_batch(batch, shuffle_vector)
        sentence_embeddings = self.SentenceTransformerModel.encode(discriminator_batch, convert_to_tensor=True)
        return self._discriminator_bert_embeddings_to_predictions(sentence_embeddings)

    def test_epoch_end(self, outputs) -> None:
        pass

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    """################# UTILS FUNCTIONS #####################"""



    def _convert_to_list_of_dicts(self, batch):
        # to make it a shape of {'origin':int,'biased':string,'unbiased':string}
        batch_df = pd.DataFrame.from_dict(batch)
        batch = batch_df.T.to_dict().values()
        return batch