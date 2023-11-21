from custom_gpt_transformers import BioGPTTransformer, GPT2MediumTransformer
from custom_losses import OnlineContrastiveLoss, ContrastiveLoss, CLMLoss, SoftMaxLoss

import random
import os

import pytz
import torch
from torch import nn
import json
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from datetime import datetime
from sentence_transformers import SentenceTransformer,  models, losses
import re
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from train_sentence_bert import *
from sklearn.metrics import roc_auc_score, accuracy_score
# import train_sentence_bert
from GAN.Utils.src.TextUtils import BreakSentenceBatch
import torch.nn.functional as F
"""PubMedGAN implementation 
"""
GENERATOR_OPTIMIZER_INDEX = 1

"""
This class implements the GAN model using SentenceTransformer as the embedding model.
"""

GPT_TRANSFORMERS = {'gpt2-medium': GPT2MediumTransformer, 'microsoft/biogpt': BioGPTTransformer}

class PubMedGanSentenceTransformer(pl.LightningModule):
    def __init__(self, hparams):
        super(PubMedGanSentenceTransformer, self).__init__()
        self.hparams.update(hparams)
        self.max_seq_length = self.hparams['max_seq_length']
        self.name = "" if not self.hparams['only_classifier_params'] else "only_classifier_params_"

        #sentence transformer based BERT impl 
        if self.hparams['base_model'] == 'all-MiniLM-L6-v2':
            self.SentenceTransformerModel = SentenceTransformer(self.hparams['base_model'])
            self.sentence_embedding_size = 384
            self.name += f"sbert_{self.hparams['base_model']}_normalized_disable_nsp_={self.hparams['disable_nsp_loss']}_disable_disc={self.hparams['disable_discriminator']}_varied={self.hparams['varied_pairs']}_sts_{self.hparams['sts_pairs']}_loss={self.hparams['loss']}"
            if self.hparams['loss'] == 'SoftMaxLoss':
                self.lm_loss = SoftMaxLoss(self.SentenceTransformerModel, self.SentenceTransformerModel.get_sentence_embedding_dimension(), num_labels=2)
            elif self.hparams['loss'] == 'OnlineContrastiveLoss':
                self.lm_loss = OnlineContrastiveLoss(losses.SiameseDistanceMetric.COSINE_DISTANCE, self.hparams['sbert_loss_margin'])
            else:
                self.lm_loss = ContrastiveLoss(losses.SiameseDistanceMetric.COSINE_DISTANCE, self.hparams['sbert_loss_margin'])

        #sentence gpt impl
        elif self.hparams['base_model'] in GPT_TRANSFORMERS.keys():
            # TODO: this is an infrastructure for loading a GPT sentence transformer model from an existing checkpoint path, it currently is broken because when we load a GPT-base sentence
            # transformer, it loads GPT without the CLM head (that we use for calculating CLM loss during generator step), need to fix this, for now don't use
            if self.hparams['checkpoint_path'] is not None:
                print(f"loading model from checkpoint path: {self.hparams['checkpoint_path']}")
                self.SentenceTransformerModel =  SentenceTransformer(self.hparams['checkpoint_path'])
                self.sentence_embedding_size = self.SentenceTransformerModel.get_sentence_embedding_dimension()
            # the checkpoint path is None only when we train from scratch
            else:
                word_embedding_model = GPT_TRANSFORMERS[self.hparams['base_model']](max_seq_length=self.max_seq_length)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
                self.SentenceTransformerModel = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                self.sentence_embedding_size = word_embedding_model.get_word_embedding_dimension()

            self.max_input_length = self.hparams['max_length_bert_input']  # for CLM task, we need to limit the number of tokens per sentence in the input from the batch, unrelated to the max_seq_length of sentence transformers
             # set the data collator for CLM task (MLM= false because we don't want to mask tokens randomly like in MLM task)
            self.data_collator = DataCollatorForLanguageModeling(self.SentenceTransformerModel.tokenizer, mlm=False)

            self.SentenceTransformerModel.max_seq_length = self.max_seq_length
            
            self.name += f"{self.hparams['base_model']}_on_clm_disable_disc={self.hparams['disable_discriminator']}"
            self.lm_loss = CLMLoss(self.SentenceTransformerModel)
        else:
            raise NotImplementedError(f"base_model {self.hparams['base_model']} not supported!")
        
        # common fields for both BERT-based and GPT-based implenetations
        self.max_epochs = self._set_number_of_epochs()
        self.model_name = self.hparams['base_model'].replace("/", "_")
        self.discriminator_loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

        # *3 because we contatenate the 2 sentences and the abs element-wise diff between them (see _discriminator_sentence_transformer_embeddings_to_predictions)
        self.classifier = nn.Linear(self.sentence_embedding_size * 3, 1)
        self.save_model_path = os.path.join(
            self.hparams['SAVE_PATH'], f"{self.model_name}_{datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f')}")
        print(self.save_model_path)
        
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
            for p in self.SentenceTransformerModel.parameters():
                p.requires_grad = True

            if self.hparams["disable_discriminator"]:
                step_ret_dict['loss'] = 0
            else: 
                step_ret_dict = self._discriminator_step(batch, name)
            if (step_ret_dict == None):
                # if there are no pairs for _discriminator_step, the output is None, but we still preform generator step
                step_ret_dict = {}
            self.SentenceTransformerModel = self.SentenceTransformerModel.to(device="cuda")
            step_ret_dict = self._generator_step(batch, step_ret_dict, name)
            step_ret_dict["step"] = "generator"

        return step_ret_dict

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        if self.hparams["disable_discriminator"]:
            return self.step(batch, GENERATOR_OPTIMIZER_INDEX, 'train_dataset')
        return self.step(batch, optimizer_idx, 'train_dataset')

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx: int = None) -> dict:
        outs = []
        if self.hparams["disable_discriminator"]:
            return self.step(batch, GENERATOR_OPTIMIZER_INDEX, 'val_dataset')
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
            # TODO: WHY? only generator outputs. TODO: really?
            outputs = outputs[0]
        losses = [output['losses'] for output in outputs if 'losses' in output]
        y_true = [output['y_true'] for output in outputs if 'y_true' in output]
        y_proba = [output['y_proba']
                   for output in outputs if 'y_proba' in output]
        y_score = [output['y_score']
                   for output in outputs if 'y_score' in output]
        if losses:
            losses = torch.cat(losses)
            self.log(f'debug/{name}_loss', losses.mean(),
                     batch_size=self.hparams['batch_size'])
        if y_true and y_proba:
            y_true = torch.cat(y_true)
            y_proba = torch.cat(y_proba)
            self.log(f'debug/{name}_accuracy', (1. * ((1. * (y_proba >= 0.5))
                     == y_true)).mean(), batch_size=self.hparams['batch_size'])
            self.log(f'debug/{name}_1_accuracy', (1. * (y_proba[y_true == 1] >= 0.5)).mean(),
                     batch_size=self.hparams['batch_size'])
            self.log(f'debug/{name}_0_accuracy', (1. * (y_proba[y_true == 0] < 0.5)).mean(),
                     batch_size=self.hparams['batch_size'])
        if name == 'val_dataset':
            path = os.path.join(self.save_model_path,
                                f"epoch_{self.current_epoch + self.inital_epoch}")
            if self.current_epoch % 5 == 0 and not os.path.exists(path):
                os.makedirs(path)
                self.SentenceTransformerModel.save(path)
                if not os.path.exists(f'{path}/config'):
                    np.save(f'{path}/config', self.hparams)
                # overwrite the type field in modules.json because some transformers we use re custom
                # but running the downstream task requires the original sentence_transformers.models.Transformer file when loading
                with open(f'{path}/modules.json', 'r+') as file:
                    data = json.load(file)
                    data[0]['type'] = 'sentence_transformers.models.Transformer'
                    file.seek(0)
                    json.dump(data, file, indent=4)



    def configure_optimizers(self):
        # we had to  set requires_grad to True due to a 'no_grad' error in the backpropagation
        for p in self.SentenceTransformerModel.parameters():
            p.requires_grad = True

        grouped_parameters_discriminator = [
            {'params': self.classifier.parameters()}]
        if not self.hparams['only_classifier_params']:
            # we let the discriminator update the SentenceTransformerModel weights only if we take this branch
            grouped_parameters_discriminator += [{'params': self.SentenceTransformerModel.parameters()}]

        optimizer_discriminator = torch.optim.Adam(
            grouped_parameters_discriminator, lr=self.hparams['learning_rate'])
        # Generator step parameters - only 'the SentenceTransformerModel parameters.
        grouped_parameters_generator = [
            {'params': self.SentenceTransformerModel.parameters()}]
        optimizer_generator = torch.optim.Adam(
            grouped_parameters_generator, lr=self.hparams['learning_rate'])
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
        discriminator_y_true = torch.as_tensor(
            [float(random.choice([0, 1])) for _ in clean_discriminator_batch])
        result_dictionary['y_true'] = discriminator_y_true
        # discriminator_y_true created in order to shuffle the bias/unbiased order
        discriminator_predictions = self._discriminator_get_predictions(
            clean_discriminator_batch, discriminator_y_true)
        all_samples_losses = self.discriminator_loss_func(
            discriminator_predictions, discriminator_y_true.to(self.device))
        discriminator_loss = all_samples_losses.mean()
        result_dictionary['loss'] = discriminator_loss
        result_dictionary['losses'] = all_samples_losses
        self.log(f'discriminator/{name}_loss', discriminator_loss,
                 batch_size=self.hparams['batch_size'])
        y_proba = self._y_pred_to_probabilities(
            discriminator_predictions).cpu().detach()
        result_dictionary['y_proba'] = y_proba
        result_dictionary['y_score'] = discriminator_y_true.cpu().detach() * y_proba + (
            1 - discriminator_y_true.cpu().detach()) * (1 - y_proba)
        if not all(discriminator_y_true) and any(discriminator_y_true):
            # Calc auc only if batch has more than one class.
            auc = roc_auc_score(discriminator_y_true.cpu().detach(), y_proba)
            result_dictionary['auc'] = auc
            self.log(f'discriminator/{name}_auc', auc,
                     batch_size=self.hparams['batch_size'])
        accuracy = accuracy_score(
            discriminator_y_true.cpu().detach(), y_proba.round())
        result_dictionary['accuracy'] = accuracy
        self.log(f'discriminator/{name}_accuracy',
                 accuracy, batch_size=self.hparams['batch_size'])

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

    def _discriminator_sentence_transformer_embeddings_to_predictions(self, sentence_embeddings):
        """ for each pair of sentences in the batch, we concat the embeddings of the 2 sentences and the abs element-wise diff between them"""
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

        discriminator_batch = self._discriminator_get_batch(
            batch, shuffle_vector)
        sentences_list_collated = self.smart_batching_collate(self.SentenceTransformerModel, discriminator_batch)
        sentence_embeddings =  self.SentenceTransformerModel(sentences_list_collated)["sentence_embedding"]
        return self._discriminator_sentence_transformer_embeddings_to_predictions(sentence_embeddings)

    def smart_batching_collate(self,model, batch):
        """ Transforms a batch to a batch of tensors for the model using the model's tokenizer"""
        tokenized = model.tokenize(batch)
        for key in tokenized.keys():
            tokenized[key] = tokenized[key].to("cuda")
        return tokenized


    """################# GENERATOR FUNCTIONS #####################"""

    def _create_unique_set_and_indices(self, prepared_batch):
        """ creates a unique set of sentences and returns the indices of the sentences in the original batch"""
        sentences_1 = list(prepared_batch["sentence_1"].values)
        sentences_2 = list(prepared_batch["sentence_2"].values)
        all_sentences = sentences_1 + sentences_2
        unique_sentences_list = list(set(all_sentences))
        sentence_1_indices = [unique_sentences_list.index(sentence) for sentence in sentences_1]
        sentences_2_indices = [unique_sentences_list.index(sentence) for sentence in sentences_2]
        return unique_sentences_list, sentence_1_indices, sentences_2_indices

    def _match_embedding_to_sentence(self, sentence_embeddings, sentence1_indices, sentence2_indices):
        """ matches the embeddings to the sentences using the indices"""
        sentence1_embeddings = sentence_embeddings[sentence1_indices]
        sentence2_embeddings = sentence_embeddings[sentence2_indices]
        return sentence1_embeddings, sentence2_embeddings

    def _generator_step(self, batch, discriminator_step_ret_dict, name):
        """ calculates the discriminator loss and subtracts it from one of the three losses (NSP/STS/CLM) of the embedding model)"""
        step_ret_dict = discriminator_step_ret_dict
        if (not step_ret_dict):
            # if the discriminator dict is empty - the discriminator batch was empty - there were no pairs
            discriminator_loss = 0
        else:
            discriminator_loss = discriminator_step_ret_dict['loss']

        # handle the case of CLM loss in a seperate function
        if self.hparams['loss'] == 'CLMLoss':
            return self._generator_step_clm(batch, step_ret_dict, discriminator_loss, name)
        
        step_ret_dict['optimizer_idx'] = 1
        # {'loss': , 'losses': , 'nsp_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        if self.hparams['disable_discriminator']:
            self.hparams['lm_task_factor'] = 1
            self.hparams['discriminator_factor'] = 0

        if self.hparams["disable_nsp_loss"]:
            self.hparams['discriminator_factor'] = 1
            nsp_loss = torch.tensor([0]).to(device="cuda")
            total_loss = - self.hparams['discriminator_factor'] * discriminator_loss

        else:
            nsp_loss = self._get_nsp_loss(batch)
            step_ret_dict['nsp_loss'] = nsp_loss
            total_loss = self.hparams['lm_task_factor'] * nsp_loss - \
                self.hparams['discriminator_factor'] * discriminator_loss

        self.log(f'generator/{name}_nsp_loss', nsp_loss.item(),
                    batch_size=self.hparams['batch_size'])
        step_ret_dict['loss'] = total_loss
        self.log(f'generator/{name}_loss', total_loss.item(),
                 batch_size=self.hparams['batch_size'])
        self.log(f'generator/{name}_discriminator_loss',
                 discriminator_loss, batch_size=self.hparams['batch_size'])
        return step_ret_dict

    def _generator_step_clm(self, batch, step_ret_dict, discriminator_loss, name):
        if self.hparams['disable_discriminator']:
            self.hparams['lm_task_factor'] = 1
            self.hparams['discriminator_factor'] = 0
        generator_batch = self._generator_get_batch(batch)
        _, documents_sentences, _ = BreakSentenceBatch(generator_batch)
        gpt_inputs = self._get_gpt_inputs(documents_sentences)
        clm_loss = self.lm_loss(gpt_inputs)
        self.log(f'generator/{name}_clm_loss', clm_loss.item(), batch_size=self.hparams['batch_size'])
        self.log(f'generator/{name}_discriminator_loss', discriminator_loss, batch_size=self.hparams['batch_size'])
        total_loss = self.hparams['lm_task_factor'] * clm_loss - \
                self.hparams['discriminator_factor'] * discriminator_loss
        self.log(f'generator/{name}_loss', total_loss.item(),batch_size=self.hparams['batch_size'])
        step_ret_dict['clm_loss'] = clm_loss
        step_ret_dict['loss'] = total_loss
        return step_ret_dict


    def _generator_get_batch(self, batch):
        result_batch = []
        for index, entry in enumerate(batch):
            result_batch.append(entry['origin_text'])
        return result_batch

    """################# UTILS FUNCTIONS #####################"""

    def _get_gpt_inputs(self, documents_sentences):
        """ used in CLM loss calculation to get the inputs and labels for the GPT model"""
         # Tokenization: Convert sentences to model-friendly format
        inputs = self.SentenceTransformerModel.tokenizer.batch_encode_plus(documents_sentences, padding=True, truncation=True,
                                                       max_length=self.max_input_length,
                                                       add_special_tokens=True, return_tensors="pt")
        # Moving the tokenized inputs to the specified device (GPU/CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        collated_inputs = self.data_collator(inputs['input_ids'].tolist())
        # Moving the collated inputs to the specified device
        collated_inputs = {k: v.to(self.device) for k, v in collated_inputs.items()}
        # Updating the input dictionary with collated 'input_ids' and 'labels'
        inputs['input_ids'] = collated_inputs['input_ids']
        inputs['labels'] = collated_inputs['labels']
        return inputs

    def _get_nsp_loss(self,batch):
        """ calculates the nsp loss using the sentence transformer model"""
        if self.hparams['varied_pairs']:
            prepared_batch = prepare_varied_batch_from_gan(batch)
        elif self.hparams['sts_pairs']:
            prepared_batch = prepare_sts_batch_from_gan(batch)
        else:
            prepared_batch = prepare_batch_from_gan(batch)
        unique_sentences_list, sentence1_indices, sentence2_indices = self._create_unique_set_and_indices(prepared_batch)
        unique_sentences_list_collated = self.smart_batching_collate(self.SentenceTransformerModel, unique_sentences_list)
        sentence_embeddings = self.SentenceTransformerModel(unique_sentences_list_collated)
        sentence1_embeddings, sentence2_embeddings = self._match_embedding_to_sentence(sentence_embeddings["sentence_embedding"], sentence1_indices, sentence2_indices)
        labels =  torch.tensor(prepared_batch["label"].values).to(device="cuda")
        return self.lm_loss([sentence1_embeddings,sentence2_embeddings],labels) # calculates the contrastive divergence loss using cosine similarity

    def _convert_to_list_of_dicts(self, batch):
        # to make it a shape of {'origin':int,'biased':string,'unbiased':string}
        batch_df = pd.DataFrame.from_dict(batch)
        batch = batch_df.T.to_dict().values()
        return batch

    def _set_number_of_epochs(self):
        """ sets the number of epochs to train according to the checkpoint path if exists, otherwise we set it to max_epochs"""
        if self.hparams['checkpoint_path'] is not None:
            #set number of epochs when given from checkpoint path to remaining epochs upto max_epochs at hparams
            epoch_match = re.search(r'epoch_(\d+)', self.hparams['checkpoint_path'])
            epoch_number = int(epoch_match.group(1))
            max_epochs = self.hparams['max_epochs'] - epoch_number
            print("Set number of epochs to: ", max_epochs)
            self.inital_epoch = epoch_number
            return max_epochs
        else:
            self.inital_epoch = 0
            return self.hparams['max_epochs']
