import os
from pathlib import Path

hparams = {'learning_rate': 2e-5,
           'dropout_rate': 0.1,
           'hidden_sizes': [512],
           'discriminator_factor': 0.5,
           'mlm_factor': 0.5,
           'nsp_factor': 0.5,
           'gpus': 1,
           'max_epochs': 20,
           'gender_and_topic_path': os.path.join(
               os.pardir,
               os.pardir,
               'data',
               'abstract_2005_2020_gender_and_topic'
           ),
           'disable_nsp_loss': False,
           'disable_discriminator': False,
           'batch_size': 2,
           'test_size': 0.3,
           'bert_pretrained_over_pubMed_path': Path(__file__).resolve().parent.parent.parent.parent.parent / 'saved_models' / 'bert_tiny_uncased_2010_2018_v2020_epoch39',
           'bert_tokenizer': os.path.join('google', 'bert_uncased_L-2_H-128_A-2'),
           'similarity_metric': 'cross_entropy',
           'cosine_similarity_clean_method': '',
           'reset_different_topic_entries_flag': 1,
           'max_length_bert_input': 50,
           'max_sentences_per_abstract': 20,
           'SAVE_PATH': Path(__file__).resolve().parent.parent.parent.parent.parent / 'saved_models' /'GAN_experiments_models' ,
           }
