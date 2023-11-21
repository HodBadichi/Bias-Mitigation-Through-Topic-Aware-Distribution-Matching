from transformers import AutoTokenizer, BertForMaskedLM, AutoModel, ElectraForPreTraining, ElectraTokenizerFast



from GAN.Discriminator.src.Discriminator import Discriminator 
import torch.nn as nn

"""
Different Discriminators that inherit from base Discrimnator class using different BERT models 
"""


class DiscriminatorBioElectra(Discriminator):
    def __init__(self, hparams):
        super(DiscriminatorBioElectra, self).__init__(hparams)
        self.bert_model = ElectraForPreTraining.from_pretrained("kamalkraj/bioelectra-base-discriminator-pubmed")
        self.bert_tokenizer = ElectraTokenizerFast.from_pretrained("kamalkraj/bioelectra-base-discriminator-pubmed")
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.name = "BioElectra over pubmed with numbers"
        self.classifier = nn.Sequential(*self._calculate_layers())


class DiscriminatorSciBert(Discriminator):
    def __init__(self, hparams):
        super(DiscriminatorSciBert, self).__init__(hparams)
        self.bert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.bert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.name = "SciBert with numbers"
        self.classifier = nn.Sequential(*self._calculate_layers())

class DiscriminatorTinyBert(Discriminator):
    def __init__(self, hparams):
        super(DiscriminatorTinyBert, self).__init__(hparams)
        self.bert_model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.name = "TinyBert over pubmed with numbers"
        self.classifier = nn.Sequential(*self._calculate_layers())

class DiscriminatorLinkBert(Discriminator):
    def __init__(self, hparams):
        super(DiscriminatorLinkBert, self).__init__(hparams)
        self.bert_model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")
        self.bert_tokenizer =  AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
        self.sentence_embedding_size = self.bert_model.get_input_embeddings().embedding_dim
        self.name = "LinkBert over pubmed with urls and numbers"
        self.classifier = nn.Sequential(*self._calculate_layers())


