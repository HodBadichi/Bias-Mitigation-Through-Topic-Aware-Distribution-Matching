import os
from pathlib import Path

hparams = {'learning_rate': 2e-5,
           'dropout_rate': 0.1,
           'hidden_sizes': [512],
           'discriminator_factor': 0.5,
           'mlm_factor': 0.5,
           'gpus': 1,
           'max_epochs': 40,
           'batch_size': 8,
           'test_size': 0.3,
           'max_length_bert_input': 50,
           'max_sentences_per_abstract': 20,
           'reset_different_topic_entries_flag': 1,
           'similarity_metric': 'cross_entropy',
           'gender_and_topic_path': os.path.join(
               os.pardir,
               os.pardir,
               'data',
               'abstract_2005_2020_gender_and_topic'
           ),
           'bert_pretrained_over_pubMed_path': Path(__file__).resolve().parents[4] / 'saved_models' / 'bert_tiny_uncased_2010_2018_v2020_epoch39',
           'bert_tokenizer': os.path.join('google', 'bert_uncased_L-2_H-128_A-2'),
           'cosine_similarity_clean_method': '',
           'SAVE_PATH': "/home/mor.filo/nlp_project/DistributionMatching/GANPubMed/Models",
           'DEBUG_FLAG': False
           }
