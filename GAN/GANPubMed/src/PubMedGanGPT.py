import random
import os

import pytz
import torch
from torch import nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from datetime import datetime
# from transformers import AutoTokenizer, gptForMaskedLM, DataCollatorForLanguageModeling
from transformers import BioGptTokenizer, BioGptForCausalLM, DataCollatorForLanguageModeling
from sklearn.metrics import roc_auc_score, accuracy_score

from GAN.Utils.src.TextUtils import BreakSentenceBatch

"""PubMedGANGPT implementation 
"""


class PubMedGANGPT(pl.LightningModule):
    def __init__(self, hparams):
        super(PubMedGANGPT, self).__init__()
        self.hparams.update(hparams)
        self.name = f"microsoft/biogpt"
        self.gpt_model = BioGptForCausalLM.from_pretrained(self.name) #should we train on casual language modeling?
        self.gpt_tokenizer =  BioGptTokenizer.from_pretrained(self.name)
        self.data_collator = DataCollatorForLanguageModeling(self.gpt_tokenizer, mlm=False) 
        self.max_length_gpt_input = self.hparams['max_length_bert_input'] #TODO: should be left as is?
        self.max_sentences_per_abstract = self.hparams['max_sentences_per_abstract']
        self.sentence_embedding_size = self.gpt_model.config.hidden_size
        # self.downsampler = nn.Linear(self.sentence_embedding_size, 384) #for now we use the same size as the sbert model
        # The linear layer if from 2 concat abstract (1 is bias and 1 unbiased) to binary label:
        # 1 - the couple of matching docs was [biased,unbiased]
        # 0 - the couple of matching docs was [unbiased,biased]
        self.classifier = nn.Linear(self.sentence_embedding_size * self.max_sentences_per_abstract * 2, 1)
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.save_model_path = os.path.join(
            self.hparams['SAVE_PATH'], f"GPT_{self.name}_{datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%d%m%y_%H%M%S.%f')}")
        os.makedirs(self.save_model_path, exist_ok=True)
        self.freeze_model_parameters()
        self.name += f"clm_loss={self.hparams['disable_clm_loss']}"

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
        # {'loss': , 'losses': , 'clm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        if optimizer_idx == 0:
            step_ret_dict = self._discriminator_step(batch, name)
        #   Generator Step
        if optimizer_idx == 1:
            step_ret_dict = self._discriminator_step(batch, name)
            if (step_ret_dict == None):
                # f there are no pairs for _discriminator_step, the output is None, but we still preform generator step
                step_ret_dict = {}
            step_ret_dict = self._generator_step(batch, step_ret_dict, name)
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
        losses = [output['losses'] for output in outputs if 'losses' in output]
        y_true  =[output['y_true'] for output in outputs if 'y_true' in output]
        y_proba =[output['y_proba'] for output in outputs if 'y_proba' in output]
        y_score =[output['y_score'] for output in outputs if 'y_score' in output]
        if losses:
            losses = torch.cat(losses)
            self.log(f'debug/{name}_loss', losses.mean(), batch_size=self.hparams['batch_size'])
        if y_true and y_proba:
            y_true = torch.cat(y_true)
            y_proba = torch.cat(y_proba)
            self.log(f'debug/{name}_accuracy', (1. * ((1. * (y_proba >= 0.5)) == y_true)).mean(),batch_size=self.hparams['batch_size'])
            self.log(f'debug/{name}_1_accuracy', (1. * (y_proba[y_true == 1] >= 0.5)).mean(),
                    batch_size=self.hparams['batch_size'])
            self.log(f'debug/{name}_0_accuracy', (1. * (y_proba[y_true == 0] < 0.5)).mean(),
                 batch_size=self.hparams['batch_size'])
        if name == 'val_dataset':
            path = os.path.join(self.save_model_path,
                                f"epoch_{self.current_epoch}")
            if self.current_epoch > 0 and not os.path.exists(path):
                os.makedirs(path)
                self.gpt_model.save_pretrained(path)
                self.gpt_tokenizer.save_pretrained(path)
                if not os.path.exists(f'{path}/config'):
                    np.save(f'{path}/config', self.hparams)

    def configure_optimizers(self):
        # Discriminator step paramteres -  classifier.
        grouped_parameters_discriminator = [{'params': self.classifier.parameters()}]
        grouped_parameters_discriminator += [
            {'params': self.gpt_model.parameters()}]
        optimizer_discriminator = torch.optim.Adam(grouped_parameters_discriminator, lr=self.hparams['learning_rate'])
        # Generator step parameters - only 'the gpt model.
        grouped_parameters_generator = [{'params': self.gpt_model.parameters()}]
        optimizer_generator = torch.optim.Adam(grouped_parameters_generator, lr=self.hparams['learning_rate'])
        return [optimizer_discriminator, optimizer_generator]

    def freeze_model_parameters(self):
        # freeze all layers
        for param in self.gpt_model.parameters():
            param.requires_grad = False
        # unfreeze last layer
        for param in self.gpt_model.biogpt.layers[-1].parameters():
            param.requires_grad = True
        # if clm loss is enabled we have to unfreeze the output_projection layer
        for param in self.gpt_model.output_projection.parameters():
            param.requires_grad = True

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
        result_dictionary = {'clm_loss': 0, 'optimizer_idx': 0}
        # {'loss': , 'losses': , 'clm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
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

    def _discriminator_get_cls_gpt_outputs(self, gpt_inputs):
        all_outputs = self.gpt_model(**gpt_inputs, output_hidden_states=True)
        cls_outputs = all_outputs.hidden_states[-1][:, 0]
        return cls_outputs

    def _discriminator_gpt_embeddings_to_predictions(self, gpt_cls_outputs, begin_end_indexes):
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
                gpt_cls_outputs[start_index_first_document: end_index_first_document])
            second_document_embeddings = fix_sentences_size(
                gpt_cls_outputs[start_index_second_document: end_index_second_document])
            curr_concat_embeddings = torch.cat((first_document_embeddings, second_document_embeddings))
            sample_embedding.append(curr_concat_embeddings)

        aggregated = torch.stack(sample_embedding)
        y_predictions = self.classifier(aggregated).squeeze(1)
        return y_predictions

    def _discriminator_get_predictions(self, batch, shuffle_vector):
        """
        :param batch: a batch of PubMedGANGPT in the shape of {'origin':int,'biased':string,'unbiased':string}
        might include `None` values in the `biased` and `unbiased` entry in case the origin document has no match.

        :return: This function wil return the classifier predictions over gptmodel output embeddings
        shuffle_vector: list in the shape of [0,...,1,0...] which can be interpreted like that:
        1 - the couple of matching docs was [biased,unbiased]
        0 - the couple of matching docs was [unbiased,biased]
        """

        discriminator_batch = self._discriminator_get_batch(batch, shuffle_vector)

        begin_end_indexes, documents_sentences, max_len = BreakSentenceBatch(discriminator_batch)

        gpt_inputs = self._get_gpt_inputs(documents_sentences)

        gpt_cls_outputs = self._discriminator_get_cls_gpt_outputs(gpt_inputs)
        return self._discriminator_gpt_embeddings_to_predictions(gpt_cls_outputs, begin_end_indexes)

    """################# GENERATOR FUNCTIONS #####################"""

    def _generator_step(self, batch, discriminator_step_ret_dict, name):
        step_ret_dict = discriminator_step_ret_dict
        if not step_ret_dict:
            if self.hparams["disable_clm_loss"]: #we can't calculate any loss if discriminator loss is not available and we don't use CLM loss
                return None
            # if the discriminator dict is empty - the discriminator batch was empty - there were no pairs
            else:
                discriminator_loss = 0
        else:
            discriminator_loss = discriminator_step_ret_dict['loss']
        step_ret_dict['optimizer_idx'] = 1
        if self.hparams['disable_clm_loss']:
            self.hparams['discriminator_factor'] = 1
            clm_loss = torch.tensor([2]).to(device="cuda")
            total_loss = - self.hparams['discriminator_factor'] * discriminator_loss
        else:
        # {'loss': , 'losses': , 'clm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
            generator_batch = self._generator_get_batch(batch)
            begin_end_indexes, documents_sentences, max_len = BreakSentenceBatch(generator_batch)
            gpt_inputs = self._get_gpt_inputs(documents_sentences)
            clm_loss = self._generator_get_clm_loss(gpt_inputs)
            step_ret_dict['clm_loss'] = clm_loss
            total_loss = self.hparams['mlm_factor'] * clm_loss - self.hparams['discriminator_factor'] * discriminator_loss
        step_ret_dict['loss'] = total_loss
        self.log(f'generator/{name}_clm_loss', clm_loss, batch_size=self.hparams['batch_size'])
        self.log(f'generator/{name}_loss', total_loss, batch_size=self.hparams['batch_size'])
        self.log(f'generator/{name}_discriminator_loss', discriminator_loss, batch_size=self.hparams['batch_size'])
        return step_ret_dict

    def _generator_get_clm_loss(self, inputs):
        """returns CLM loss"""
        collated_inputs = self.data_collator(inputs['input_ids'].tolist())
        collated_inputs = {k: v.to(self.device) for k, v in collated_inputs.items()}

        inputs['input_ids'] = collated_inputs['input_ids']
        inputs['labels'] = collated_inputs['labels']
        vals = self.gpt_model(**inputs)
        loss = vals.loss
        return loss

    def _generator_get_batch(self, batch):
        result_batch = []
        for index, entry in enumerate(batch):
            result_batch.append(entry['origin_text'])
        return result_batch

    """################# UTILS FUNCTIONS #####################"""
    

    def _get_gpt_inputs(self, documents_sentences):
        inputs = self.gpt_tokenizer.batch_encode_plus(documents_sentences, padding=True, truncation=True,
                                                       max_length=self.max_length_gpt_input,
                                                       add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _convert_to_list_of_dicts(self, batch):
        # to make it a shape of {'origin':int,'biased':string,'unbiased':string}
        batch_df = pd.DataFrame.from_dict(batch)
        batch = batch_df.T.to_dict().values()
        return batch