import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, BertForMaskedLM
from pytorch_lightning import Trainer
from DistributionMatching.utils import config
import numpy as np
import random
from sklearn.metrics import roc_auc_score, accuracy_score



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

        # 0 : [biased,unbiased]
        # 1 : [unbiased,biased]
        # we`ll use y_true for loss and also in order to shuffle 'biased','unbiased' tuple
        y_true = [random.choice([0, 1]) for _ in batch]

        # `discriminator_loss` is used in both Discriminator and Generator losses
        discriminator_predictions = self._get_discriminator_predictions(batch, y_true)
        all_samples_losses = self.loss_func(discriminator_predictions, y_true)
        discriminator_loss = all_samples_losses.mean(all_samples_losses)

        #   Discriminator Step
        if optimizer_idx == 0:
            mlm_loss = 0
            self.log(f'discriminator/{name}_loss', discriminator_loss)
            y_proba = self.y_pred_to_probabilities(discriminator_predictions).cpu().detach()

            if not all(y_true) and any(y_true):
                # Calc auc only if batch has more than one class.
                self.log(f'discriminator/{name}_auc', roc_auc_score(y_true.cpu().detach(), y_proba))
            self.log(f'discriminator/{name}_accuracy', accuracy_score(y_true.cpu().detach(), y_proba.round()))

        #   Generator Step
        if optimizer_idx == 1:
            generator_batch = self._get_generator_batch(batch)
            begin_end_indexes, documents_sentences, max_len = self._break_sentence_batch(generator_batch)
            bert_inputs = self._get_bert_inputs(documents_sentences)

            mlm_loss = self._get_mlm_loss(bert_inputs)
            diff_frozen_lost = self._get_diff_frozen(bert_inputs)

            y_proba = self.y_pred_to_probabilities(discriminator_predictions).cpu().detach()
            losses = self.loss_func(discriminator_predictions, 1 - y_true)
            d_loss = losses.mean()
            loss =


            self.log(f'generator/{name}_loss', loss)
            self.log(f'generator/{name}_mlm_loss', mlm_loss)
            self.log(f'generator/{name}_frozen_diff_loss', diff_frozen_lost)
            self.log(f'generator/{name}_discriminator_loss', d_loss)

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def configure_optimizers(self):
        pass

    def _get_discriminator_batch(self, batch, shuffle_vector):
        clean_batch = []  # batch where each document has a pair , no unmatched documents allowed
        for sample in batch:
            if sample['biased'] is None or sample['unbiased'] is None:
                continue
            clean_batch.append(sample)

        result_batch = []
        for index, entry in enumerate(clean_batch):
            biased_text = entry['biased']
            unbiased_text = entry['unbiased']
            if shuffle_vector[index] == 0:
                result_batch.append(biased_text)
                result_batch.append(unbiased_text)
            if shuffle_vector[index] == 1:
                result_batch.append(unbiased_text)
                result_batch.append(biased_text)
        return result_batch

    def _get_generator_batch(self, batch):
        result_batch = []
        for index, entry in enumerate(batch):
            result_batch.append(entry['origin_text'])
        return result_batch

    def _get_bert_outputs(self, bert_inputs):
        all_outputs = self.bert_model(**bert_inputs, output_hidden_states=True).hidden_states[-1]
        cls_outputs = all_outputs[:, 0]
        return all_outputs, cls_outputs

    def _get_bert_inputs(self, documents_sentences):
        inputs = self.bert_tokenizer.batch_encode_plus(documents_sentences, padding=True, truncation=True,
                                                       max_length=self.max_len,
                                                       add_special_tokens=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def _bertEmbeddingToPredictions(self, bert_cls_outputs, begin_end_indexes):
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

    def _get_discriminator_predictions(self, batch, shuffle_vector):
        """

        :param batch: a batch of PubMedGan in the sahpe of {'origin':int,'biased':string,'unbiased':string}
        might include `None` values in the `biased` and `unbiased` entry in case the origin document has no match.

        :param shuffle_vector: list in the shape of [0,...,1,0...]. we use each entry to tell how to concat
        the fields `biased` and `unbiased` where
        # 0 : [biased,unbiased]
        # 1 : [unbiased,biased]

        :return:This function wil return the classifier predictions over bertmodel output embeddings
        """

        discriminator_batch = self._get_discriminator_batch(batch, shuffle_vector)

        begin_end_indexes, documents_sentences, max_len = self._break_sentence_batch(discriminator_batch)

        bert_inputs = self._get_bert_inputs(documents_sentences)

        # `bert_all_outputs` is not being used.
        bert_all_outputs, bert_cls_outputs = self._get_bert_outputs(bert_inputs)
        return self._bertEmbeddingToPredictions(bert_cls_outputs, begin_end_indexes)

    def _break_sentence_batch(self, batch):
        indexes = []
        all_sentences = []
        index = 0
        max_len = 0
        for sample in batch:
            sample_as_list = sample.split('<BREAK>')
            # sample is a list of sentences
            indexes.append((index, index + len(sample_as_list)))
            index += len(sample_as_list)
            all_sentences.extend(sample_as_list)
            if max_len < len(sample_as_list):
                max_len = len(sample_as_list)
        return indexes, all_sentences, max_len

    def _get_mlm_loss(self,inputs):
        """returns MLM loss"""
        collated_inputs = self.data_collator(inputs['input_ids'].tolist())
        collated_inputs = {k: v.to(self.device) for k, v in collated_inputs.items()}

        inputs['input_ids'] = collated_inputs['input_ids']
        inputs['labels'] = collated_inputs['labels']
        loss = self.bert_model(**inputs).loss

        return loss

    def _get_diff_frozen(self,bert_inputs):
        outputs = self.bert_model(**bert_inputs, output_hidden_states=True).hidden_states[-1]
        reference_outputs = self.frozen_bert_model(**bert_inputs, output_hidden_states=True).hidden_states[-1]
        return torch.norm(outputs - reference_outputs, 2)