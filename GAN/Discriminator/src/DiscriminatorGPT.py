import random
import torch

from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import pytorch_lightning as pl
from torch import nn
from transformers import BioGptTokenizer, BioGptForCausalLM, DataCollatorForLanguageModeling, GPT2Tokenizer, GPT2Model

from GAN.Utils.src.TextUtils import BreakSentenceBatch

"""Discriminator Implementation
This class inherits from 'pl.LightningModule', A basic network with linear layers using RELU between each layer
which tries to detect which one of a documents duo given is a biased and an unbiased one based on Bert embeddings
trained over the whole PUBMED dataset
"""


class DiscriminatorGPT(pl.LightningModule):
    def __init__(self, hparams):
        super(DiscriminatorGPT, self).__init__()
        self.hparams.update(hparams)
        if self.hparams["gpt_model"] == "gpt2-medium":
            self.name = f"freezed-gpt2-medium"
            self.gpt_model = GPT2Model.from_pretrained(self.name)
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(self.name)
            if self.gpt_tokenizer.pad_token is None:
                self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        elif self.hparams["gpt_model"] == "biogpt":
            self.name = f"freezed-microsoft/biogpt"
            self.gpt_model = BioGptForCausalLM.from_pretrained(self.name)
            self.gpt_tokenizer =  BioGptTokenizer.from_pretrained(self.name)
        self.max_length_gpt_input = self.hparams['max_length_bert_input']
        self.sentence_embedding_size = self.gpt_model.config.hidden_size
        self.max_sentences_per_abstract = self.hparams['max_sentences_per_abstract']
        # The linear layer if from 2 concat abstract (1 is bias and 1 unbiased) to binary label:
        # 1 - the couple of matching docs was [biased,unbiased]
        # 0 - the couple of matching docs was [unbiased,biased]
        self._freeze_model_parameters()
        self.input_dropout = nn.Dropout(p=self.hparams['dropout_rate'])
        self.classifier = nn.Sequential(*self._calculate_layers())  # per il flatten
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self):
        # Forward is unneeded , GaN model will not infer in the future
        pass

    def step(self, batch: dict, name='train_dataset'):
        """
        :param batch:{'origin_text':list[string],'biased':list[string],'unbiased':list[string]}
        :param name:string, 'train' , 'test' or 'val'
        :return:dictionary,
        step_ret_dict {'loss': , 'losses': , 'mlm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        """
        batch = self._convert_to_list_of_dicts(batch)
        #   Discriminator Step
        # {'loss': , 'losses': , 'mlm_loss': , 'y_true': , 'y_proba': , 'y_score': , 'optimizer_idx': }
        step_ret_dict = self._discriminator_step(batch, name)
        return step_ret_dict

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, 'train_dataset')

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, 'val_dataset')

    def test_step(self, batch: dict, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # Discriminator step parameters -  classifier.
        grouped_parameters_discriminator = [{'params': self.classifier.parameters()}]
        optimizer_discriminator = torch.optim.Adam(grouped_parameters_discriminator, lr=self.hparams['learning_rate'])
        return [optimizer_discriminator]

    def _freeze_model_parameters(self):
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

    def _calculate_layers(self):
        layers = []
        hidden_sizes = [self.sentence_embedding_size * self.max_sentences_per_abstract * 2] + self.hparams[
            'hidden_sizes'] + [1]
        for i in range(len(hidden_sizes) - 1):
            layers.extend(
                [nn.Linear(hidden_sizes[i],
                           hidden_sizes[i + 1]),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Dropout(self.hparams['dropout_rate'])])
        return layers

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

    @staticmethod
    def _discriminator_clean_batch(batch):
        """
        Disposes samples from batch which does not contain two documents,
        happens if document `i` does not have a match.

        :param batch:{'origin_text':string,'biased':string,'unbiased':string}
        """
        clean_batch = []  # batch where each document has a pair , no unmatched documents allowed
        for sample in batch:
            if sample['biased'] == "" or sample['unbiased'] == "":
                continue
            clean_batch.append(sample)
        return clean_batch

    @staticmethod
    def _discriminator_get_batch(batch, shuffle_vector):
        """
        Disposes samples from batch which does not contain two documents,
        happens if document `i` does not have a match.

        :param batch:{'origin_text':string,'biased':string,'unbiased':string}
        :param shuffle_vector:Tensor, made of 0`s and 1`s each entry determines whether the ith sample  ,
        which consists  of two documents, will be ordered ['biased','unbiased'] or ['unbiased','biased']
        """
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
        """
        Infer bert outputs and return for the CLS`s  ONLY to be used later on
        :param bert_inputs: bert input
        """
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
        :param batch: a batch of PubMedGan in the shape of {'origin':int,'biased':string,'unbiased':string}
        might include `None` values in the `biased` and `unbiased` entry in case the origin document has no match.

        :return: This function wil return the classifier predictions over bertmodel output embeddings
        shuffle_vector: list in the shape of [0,...,1,0...] which can be interpreted like that:
        1 - the couple of matching docs was [biased,unbiased]
        0 - the couple of matching docs was [unbiased,biased]
        """

        discriminator_batch = self._discriminator_get_batch(batch, shuffle_vector)

        begin_end_indexes, documents_sentences, max_len = BreakSentenceBatch(discriminator_batch)

        gpt_inputs = self._get_gpt_inputs(documents_sentences)

        gpt_cls_outputs = self._discriminator_get_cls_gpt_outputs(gpt_inputs)
        return self._discriminator_gpt_embeddings_to_predictions(gpt_cls_outputs, begin_end_indexes)

    def test_epoch_end(self, outputs) -> None:
        pass
        # self._get_mean_accuracy(outputs, "test")

    def training_epoch_end(self, outputs):
        pass
        # self._get_mean_accuracy(outputs, "train")

    def validation_epoch_end(self, outputs):
        pass
        # self._get_mean_accuracy(outputs, "val")

    """################# UTILS FUNCTIONS #####################"""

    def _get_gpt_inputs(self, documents_sentences):
        assert (len(documents_sentences) > 0)
        inputs = self.gpt_tokenizer.batch_encode_plus(documents_sentences, padding=True, truncation=True,
                                                       max_length=self.max_length_gpt_input,
                                                       add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs
    
    # def _get_mean_accuracy(self, outputs, name):
    #     """
    #     :param outputs: list of dictionaries from epoch
    #     :param name: name is either "test", "train" or "val"
    #     This function will log to wandb the mean $name accuracy of the epoch
    #     """
    #     accuracy_score_total = 0
    #     for output in outputs:
    #         accuracy_score_total += output['accuracy_score']
    #     avg_accuacry = accuracy_score_total / len(outputs)
    #     self.log(f'discriminator/{name}_accuracy_score_per_epoch', avg_accuacry, on_epoch=True, prog_bar=True)

    @staticmethod
    def _convert_to_list_of_dicts(batch):
        # to make it a shape of {'origin':int,'biased':string,'unbiased':string}
        batch_df = pd.DataFrame.from_dict(batch)
        batch = batch_df.T.to_dict().values()
        return batch
