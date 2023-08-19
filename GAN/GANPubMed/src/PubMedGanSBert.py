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
from train_sentence_bert import *
from sklearn.metrics import roc_auc_score, accuracy_score
# import train_sentence_bert
from GAN.Utils.src.TextUtils import BreakSentenceBatch
import torch.nn.functional as F
"""PubMedGAN implementation 
"""

GENERATOR_OPTIMIZER_INDEX = 1

# from typing import Iterable, Dict
# import torch.nn.functional as F
# from torch import nn, Tensor
# from .ContrastiveLoss import SiameseDistanceMetric
# from sentence_transformers.SentenceTransformer import SentenceTransformer


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss. Similar to ConstrativeLoss, but it selects hard positive (positives that are far apart)
    and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
    better performances than  ConstrativeLoss.

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    Example::

        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.OnlineContrastiveLoss(model=model)

        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)
    """

    def __init__(self,  distance_metric, margin: float = 0.5):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_embeddings, labels, size_average=False):
        embeddings = [sentence_feature for sentence_feature in sentence_embeddings]

        distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss

class PubMedGANSBert(pl.LightningModule):
    def __init__(self, hparams):
        super(PubMedGANSBert, self).__init__()
        MODEL = 'all-MiniLM-L6-v2'
        self.hparams.update(hparams)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            self.hparams['bert_tokenizer'])
        self.data_collator = DataCollatorForLanguageModeling(
            self.bert_tokenizer)
        # self.max_length_bert_input = self.hparams['max_length_bert_input']
        # self.max_sentences_per_abstract = self.hparams['max_sentences_per_abstract']
        self.SentenceTransformerModel = SentenceTransformer(MODEL)
        self.SentenceTransformerModel.max_seq_length = 128
        self.sentence_embedding_size = 384
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.sbert_loss = OnlineContrastiveLoss(losses.SiameseDistanceMetric.COSINE_DISTANCE,0.5)
        # *3 because of the number of inputs in the batch
        self.classifier = nn.Linear(self.sentence_embedding_size * 3, 1)
        self.save_model_path = os.path.join(
            self.hparams['SAVE_PATH'], f"Sbert_{MODEL}_{datetime.now(pytz.timezone('Asia/Jerusalem')).strftime('%y%m%d_%H%M%S.%f')}")
        self.name = f"sbert_disable_nsp={self.hparams['disable_nsp_loss']}_disable_disc={self.hparams['disable_discriminator']}_{MODEL}"
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
            if self.hparams["disable_discriminator"]:
                step_ret_dict['loss'] = 0
            else: 
                step_ret_dict = self._discriminator_step(batch, name)
            if (step_ret_dict == None):
                # if there are no pairs for _discriminator_step, the output is None, but we still preform generator step
                step_ret_dict = {}
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
        self._get_mean_from_outputs(outputs, name)
        losses = [output['losses'] for output in outputs if 'losses' in output]
        y_true = [output['y_true'] for output in outputs if 'y_true' in output]
        y_proba = [output['y_proba']
                   for output in outputs if 'y_proba' in output]
        y_score = [output['y_score']
                   for output in outputs if 'y_score' in output]
        if losses:
            losses = torch.cat(losses)
            # TODO nofar and liel: understand what's the difference between this loss and the loss in the discriminator step - what are the graphs we expect to see?
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
                                f"epoch_{self.current_epoch}")
            if self.current_epoch > 0 and not os.path.exists(path):
                os.makedirs(path)
                self.SentenceTransformerModel.save(path)
                if not os.path.exists(f'{path}/config'):
                    np.save(f'{path}/config', self.hparams)

    def configure_optimizers(self):
        # Discriminator step paramteres -  classifier.
        for p in self.SentenceTransformerModel.parameters():
            p.requires_grad = True

        grouped_parameters_discriminator = [
            {'params': self.classifier.parameters()}]
        optimizer_discriminator = torch.optim.Adam(
            grouped_parameters_discriminator, lr=self.hparams['learning_rate'])
        # Generator step parameters - only 'the bert model.
        # grouped_parameters_generator = [{'params': self.bert_model.parameters()}]
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
        for p in self.SentenceTransformerModel.parameters():
            p.requires_grad = True

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
        all_samples_losses = self.loss_func(
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

        discriminator_batch = self._discriminator_get_batch(
            batch, shuffle_vector)
        sentence_embeddings = self.SentenceTransformerModel.encode(
            discriminator_batch, convert_to_tensor=True)
        # print(f"embeddings lengths is {len(sentence_embeddings[0])}")
        return self._discriminator_SBERT_embeddings_to_predictions(sentence_embeddings)

    """################# GENERATOR FUNCTIONS #####################"""
    def smart_batching_collate(self,model, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = model.tokenize(texts[idx])
            for key in tokenized.keys():
                tokenized[key] = tokenized[key].to("cuda")
            sentence_features.append(tokenized)

        return sentence_features, labels

    def smart_batching_collate_ours(self,model, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        tokenized = model.tokenize(list(batch))
        for key in tokenized.keys():
            tokenized[key] = tokenized[key].to("cuda")
        return tokenized

    def _create_set_from_prepared_batch(self, prepared_batch):
        sentences_1 = list(prepared_batch["sentence_1"].values)
        sentences_2 = list(prepared_batch["sentence_2"].values)
        sentences_1.extend(sentences_2)
        return set(sentences_1)

    def _match_embedding_to_sentence(self, prepared_batch, sentence_embeddings, unique_sentences_set):
        sentence1_embeddings = []
        sentence2_embeddings = []
        labels = []
        unique_sentences_set = list(unique_sentences_set)
        for index, row in prepared_batch.iterrows():
            sentence1 = row['sentence_1']
            sentence2 = row['sentence_2']
            if sentence1 in unique_sentences_set and sentence2 in unique_sentences_set:
                sentence1_embeddings.append(sentence_embeddings[unique_sentences_set.index(sentence1)])
                sentence2_embeddings.append(sentence_embeddings[unique_sentences_set.index(sentence2)])
                labels.append(row['label'])        
        return torch.stack(sentence1_embeddings),torch.stack(sentence2_embeddings), torch.tensor(labels)

    def _generator_step(self, batch, discriminator_step_ret_dict, name):

        # distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
        # margin = 0.5
        self.SentenceTransformerModel = self.SentenceTransformerModel.to(device="cuda")
        # self.SentenceTransformerModel.tokenizer = self.SentenceTransformerModel.tokenizer.to(device="cuda")
        for p in self.SentenceTransformerModel.parameters():
            p.requires_grad = True
        # train_loss = losses.OnlineContrastiveLoss(
            # model=self.SentenceTransformerModel, distance_metric=distance_metric, margin=margin)
# 
        step_ret_dict = discriminator_step_ret_dict
        if (not step_ret_dict):
            # if the discriminator dict is empty - the discriminator batch was empty - there were no pairs
            discriminator_loss = 0
        else:
            discriminator_loss = discriminator_step_ret_dict['loss']
        step_ret_dict['optimizer_idx'] = 1
        # {'loss': , 'losses': , 'nsp_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        if self.hparams["disable_nsp_loss"]:
            self.hparams['nsp_factor'] = 0
            self.hparams['discriminator_factor'] = 1
        if self.hparams['disable_discriminator']:
            self.hparams['nsp_factor'] = 1
            self.hparams['discriminator_factor'] = 0
        prepared_batch = prepare_batch_from_gan(batch)
        unique_sentences_set = self._create_set_from_prepared_batch(prepared_batch)
        # prepared_batch = prepare_varied_batch_from_gan(batch)
        train_examples = train_dataset_to_input_examples(prepared_batch)
        feat,labels = self.smart_batching_collate(self.SentenceTransformerModel,train_examples) # tokenizes the training_examples into two columns of features, one per "side", and the labels
        # sentence1_embeddings = self.SentenceTransformerModel(feat[0]) # encodes the embedding of the "left" side with forward
        # sentence2_embeddings = self.SentenceTransformerModel(feat[1]) # encodes the embedding of the "right" side with forward

        unique_sentences_set_collated = self.smart_batching_collate_ours(self.SentenceTransformerModel, unique_sentences_set)
        sentence_embeddings = self.SentenceTransformerModel(unique_sentences_set_collated)
        
        sentence1_embeddings,sentence2_embeddings, labels = self._match_embedding_to_sentence(prepared_batch, sentence_embeddings["sentence_embedding"], unique_sentences_set)

        nsp_loss = self.sbert_loss([sentence1_embeddings,sentence2_embeddings],labels) # calculates the contrastive divergence loss using cosine similarity
        # nsp_loss = train_loss(feat,labels)
        # nsp_loss = generate_nsp_loss_frotrm_batch(
            # batch, self.SentenceTransformerModel)
        
        step_ret_dict['nsp_loss'] = nsp_loss
        self.log(f'generator/{name}_nsp_loss', nsp_loss.item(),
                    batch_size=self.hparams['batch_size'])
        total_loss = self.hparams['nsp_factor'] * nsp_loss - \
            self.hparams['discriminator_factor'] * discriminator_loss
        step_ret_dict['loss'] = total_loss
        self.log(f'generator/{name}_loss', total_loss.item(),
                 batch_size=self.hparams['batch_size'])
        self.log(f'generator/{name}_discriminator_loss',
                 discriminator_loss, batch_size=self.hparams['batch_size'])
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
            accuracy_and_auc_results['accuracy'] = [
                output['accuracy'] for output in outputs]
            accuracy_and_auc_results['auc'] = [output['auc']
                                               for output in outputs if 'auc' in output]
        except:
            print(f"\n name is {name} and outputs is {outputs}\n")
            return
        # print(f"\nname is {name} auc is {accuracy_and_auc_results['auc']} and accuracy is {accuracy_and_auc_results['accuracy']}\n")
        accuracy_mean = np.mean(accuracy_and_auc_results['accuracy'])
        self.log(f'discriminator/{name}_accuracy_score_per_epoch',
                 accuracy_mean, on_epoch=True, prog_bar=True)
        if accuracy_and_auc_results['auc']:
            auc_mean = np.mean(accuracy_and_auc_results['auc'])
            self.log(f'discriminator/{name}_auc_score_per_epoch',
                     auc_mean, on_epoch=True, prog_bar=True)

    def _convert_to_list_of_dicts(self, batch):
        # to make it a shape of {'origin':int,'biased':string,'unbiased':string}
        batch_df = pd.DataFrame.from_dict(batch)
        batch = batch_df.T.to_dict().values()
        return batch
