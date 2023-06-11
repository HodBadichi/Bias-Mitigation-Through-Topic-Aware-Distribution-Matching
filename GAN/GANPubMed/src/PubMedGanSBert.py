import random
import os

import pytz
import torch
from torch import nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from train_sentence_bert import generate_nsp_loss_from_batch
from sklearn.metrics import roc_auc_score, accuracy_score

from GAN.Utils.src.TextUtils import BreakSentenceBatch

"""PubMedGAN implementation 
"""


class PubMedGANSBert(pl.LightningModule):
    def __init__(self, hparams):
        super(PubMedGANSBert, self).__init__()
        MODEL = 'all-MiniLM-L6-v2'
        self.hparams.update(hparams)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.hparams['bert_tokenizer'])
        self.data_collator = DataCollatorForLanguageModeling(self.bert_tokenizer)
        # self.max_length_bert_input = self.hparams['max_length_bert_input']
        # self.max_sentences_per_abstract = self.hparams['max_sentences_per_abstract']
        self.SentenceTransformerModel = SentenceTransformer(MODEL)
        self.sentence_embedding_size = 384
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.classifier = nn.Linear(self.sentence_embedding_size * 3, 1) # *3 because of the number of inputs in the batch
        self.save_model_path = os.path.join(self.hparams['SAVE_PATH'], f"Sbert_{MODEL}_{datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f')}") 
        self.name = f"sbert_{MODEL}"
        os.makedirs(self.save_model_path, exist_ok=True)

    def forward(self):
        # Forward is unneeded , GaN model will not infer in the future
        pass

    def step(self, batch: dict, optimizer_idx: int = None, name='train_dataset'):
        """
        :param batch:{'origin_text':list[string],'biased':list[string],'unbiased':list[string]}
        :param optimizer_idx: determines which step is it - discriminator or generator
        :param name:
        :return:
        """
        #   Discriminator Step
        batch = self._convert_to_list_of_dicts(batch)
        step_ret_dict = {}
        # {'loss': , 'losses': , 'nsp_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        if optimizer_idx == 0:
            step_ret_dict = self._discriminator_step(batch, name)
            if step_ret_dict:
                step_ret_dict["step"] = "discriminator"
        #   Generator Step
        if optimizer_idx == 1:
            step_ret_dict = self._discriminator_step(batch, name)
            if (step_ret_dict == None):
                # f there are no pairs for _discriminator_step, the output is None, but we still preform generator step
                step_ret_dict = {}
            step_ret_dict = self._generator_step(batch, step_ret_dict, name)
            step_ret_dict["step"] = "generator"

        return step_ret_dict
    
    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        return self.step(batch, optimizer_idx, 'train_dataset')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        outs = []
        for i in range(len(self.optimizers())):
            outs.append(self.step(batch, i, 'val_dataset'))
        # return value doesn't meter, its only pl requirement, the logging is with wandb
        return outs[1]  # generator

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None):
        return self.validation_step(batch, batch_idx, optimizer_idx)

    def test_epoch_end(self, outputs) -> None:
        self.on_end(outputs, 'test_dataset')

    def training_epoch_end(self, outputs):
        self.on_end(outputs, 'train_dataset')

    def validation_epoch_end(self, outputs):
        self.on_end(outputs, 'val_dataset')

    def on_end(self, outputs, name):
        # outputs is a list (len=number of batches) of dicts (as returned from the step methods).
        if name == 'train_dataset':
            outputs = outputs[0]  # TODO: WHY? only generator outputs. TODO: really?
        self._get_mean_from_outputs(outputs, name)
        losses = [output['losses'] for output in outputs if 'losses' in output]
        y_true  =[output['y_true'] for output in outputs if 'y_true' in output]
        y_proba =[output['y_proba'] for output in outputs if 'y_proba' in output]
        y_score =[output['y_score'] for output in outputs if 'y_score' in output]
        if losses:
            losses = torch.cat(losses)
            self.log(f'debug/{name}_loss', losses.mean(), batch_size=self.hparams['batch_size']) #TODO nofar and liel: understand what's the difference between this loss and the loss in the discriminator step - what are the graphs we expect to see?
        if y_true and y_proba:
            y_true = torch.cat(y_true)
            y_proba = torch.cat(y_proba)
            self.log(f'debug/{name}_accuracy', (1. * ((1. * (y_proba >= 0.5)) == y_true)).mean(),batch_size=self.hparams['batch_size'])
            self.log(f'debug/{name}_1_accuracy', (1. * (y_proba[y_true == 1] >= 0.5)).mean(),
                    batch_size=self.hparams['batch_size'])
            self.log(f'debug/{name}_0_accuracy', (1. * (y_proba[y_true == 0] < 0.5)).mean(),
                 batch_size=self.hparams['batch_size'])
        if name == 'val_dataset':
            path = os.path.join(self.save_model_path, f"epoch_{self.current_epoch}")
            if self.current_epoch > 0  and not os.path.exists(path):
                os.makedirs(path)
                self.SentenceTransformerModel.save(path)
                if not os.path.exists(f'{path}/config'):
                    np.save(f'{path}/config', self.hparams)


    def configure_optimizers(self):
        # Discriminator step paramteres -  classifier.
        grouped_parameters_discriminator = [{'params': self.classifier.parameters()}]
        optimizer_discriminator = torch.optim.Adam(grouped_parameters_discriminator, lr=self.hparams['learning_rate'])
        # Generator step parameters - only 'the bert model.
        # grouped_parameters_generator = [{'params': self.bert_model.parameters()}]
        grouped_parameters_generator = [{'params': self.SentenceTransformerModel.parameters()}]
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
        result_dictionary = {'nsp_loss': 0, 'optimizer_idx': 0}
        # {'loss': , 'losses': , 'mlm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        clean_discriminator_batch = self._discriminator_clean_batch(batch)
        if len(clean_discriminator_batch) == 0:
            # if there are no pairs for _discriminator_step, the output is None
            return None
        discriminator_y_true = torch.as_tensor([float(random.choice([0, 1])) for _ in clean_discriminator_batch])
        result_dictionary['y_true'] = discriminator_y_true
        # discriminator_y_true created in order to shuffle the bias/unbiased order
        discriminator_predictions = self._discriminator_get_predictions(clean_discriminator_batch, discriminator_y_true) 
        all_samples_losses = self.loss_func(discriminator_predictions, discriminator_y_true.to(self.device))
        discriminator_loss = all_samples_losses.mean()
        result_dictionary['loss'] = discriminator_loss
        result_dictionary['losses'] = all_samples_losses
        self.log(f'discriminator/{name}_loss', discriminator_loss, batch_size=self.hparams['batch_size'])
        y_proba = self._y_pred_to_probabilities(discriminator_predictions).cpu().detach()
        result_dictionary['y_proba'] = y_proba
        result_dictionary['y_score'] = discriminator_y_true.cpu().detach() * y_proba + (
                1 - discriminator_y_true.cpu().detach()) * (1 - y_proba) 
        if not all(discriminator_y_true) and any(discriminator_y_true):
            # Calc auc only if batch has more than one class.
            auc = roc_auc_score(discriminator_y_true.cpu().detach(), y_proba)
            result_dictionary['auc'] = auc
            self.log(f'discriminator/{name}_auc', auc, batch_size=self.hparams['batch_size'])
        accuracy = accuracy_score(discriminator_y_true.cpu().detach(), y_proba.round())
        result_dictionary['accuracy'] = accuracy
        self.log(f'discriminator/{name}_accuracy', accuracy, batch_size=self.hparams['batch_size'])

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
    
    def _discriminator_SBERT_embeddings_to_predictions(self, sentence_embeddings):
        sample_embedding = []

        for i in range(0, len(sentence_embeddings), 2):
            first_document_embeddings = sentence_embeddings[i]
            second_document_embeddings = sentence_embeddings[i + 1]
            curr_concat_embeddings = torch.cat((
                first_document_embeddings,
                second_document_embeddings,
                abs(first_document_embeddings - second_document_embeddings))
            )
            sample_embedding.append(curr_concat_embeddings)

        aggregated = torch.stack(sample_embedding)
        y_predictions = self.classifier(aggregated).squeeze(1)
        # print(f"prediction is {len(y_predictions)}")
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
        # print(f"embeddings lengths is {len(sentence_embeddings[0])}")
        return self._discriminator_SBERT_embeddings_to_predictions(sentence_embeddings)

    """################# GENERATOR FUNCTIONS #####################"""

    def _generator_step(self, batch, discriminator_step_ret_dict, name):
        step_ret_dict = discriminator_step_ret_dict
        if (not step_ret_dict):
            # if the discriminator dict is empty - the discriminator batch was empty - there were no pairs
            discriminator_loss = 0
        else:
            discriminator_loss = discriminator_step_ret_dict['loss']
        step_ret_dict['optimizer_idx'] = 1
        # {'loss': , 'losses': , 'nsp_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        nsp_loss = generate_nsp_loss_from_batch(batch, self.SentenceTransformerModel)
        step_ret_dict['nsp_loss'] = nsp_loss
        # TODO diff from frozen and tune the factors (mlm_loss is 2-5, discriminator_loss is ~0.5-1)
        total_loss = self.hparams['nsp_factor'] * nsp_loss - self.hparams['discriminator_factor'] * discriminator_loss
        step_ret_dict['loss'] = total_loss
        self.log(f'generator/{name}_loss', total_loss, batch_size=self.hparams['batch_size'])
        self.log(f'generator/{name}_nsp_loss', nsp_loss, batch_size=self.hparams['batch_size'])
        self.log(f'generator/{name}_discriminator_loss', discriminator_loss, batch_size=self.hparams['batch_size'])
        return step_ret_dict

    def _generator_get_batch(self, batch):
        result_batch = []
        for index, entry in enumerate(batch):
            result_batch.append(entry['origin_text'])
        return result_batch

    """################# UTILS FUNCTIONS #####################"""

    def _get_mean_from_outputs(self, outputs, name):
        """
        :param outputs: list of dictionaries from epoch
        :param name: name is either "test", "train" or "val"
        This function will log to wandb the mean $name accuracy/auc of the epoch
        """
        accuracy_and_auc_results = {}
        try:
            accuracy_and_auc_results['accuracy'] = [output['accuracy'] for output in outputs]
            accuracy_and_auc_results['auc'] = [output['auc'] for output in outputs if 'auc' in output]
        except:
            print(f"\n name is {name} and outputs is {outputs}\n")
            return
        # print(f"\nname is {name} auc is {accuracy_and_auc_results['auc']} and accuracy is {accuracy_and_auc_results['accuracy']}\n")
        accuracy_mean = np.mean(accuracy_and_auc_results['accuracy'])
        self.log(f'discriminator/{name}_accuracy_score_per_epoch', accuracy_mean, on_epoch=True, prog_bar=True)    
        if accuracy_and_auc_results['auc']:    
            auc_mean = np.mean(accuracy_and_auc_results['auc'])
            self.log(f'discriminator/{name}_auc_score_per_epoch', auc_mean, on_epoch=True, prog_bar=True)

    def _convert_to_list_of_dicts(self, batch):
        # to make it a shape of {'origin':int,'biased':string,'unbiased':string}
        batch_df = pd.DataFrame.from_dict(batch)
        batch = batch_df.T.to_dict().values()
        return batch


