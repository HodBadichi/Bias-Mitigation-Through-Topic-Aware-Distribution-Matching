import os
from pathlib import Path

hparams = {'learning_rate': 3e-5,
           'dropout_rate': 0.1,
           'hidden_sizes': [512],
           'discriminator_factor': 0.5,
           'max_seq_length' : 128,
           'mlm_factor': 0.5,
           'lm_task_factor': 0.5,
           'gpus': 1,
           'max_epochs': 30,
           'gender_and_topic_path': str(Path(__file__).resolve().parents[2] / 'data' / 'abstract_2005_2022_gender_and_topic'),
           'splitted_PubMedData':  Path(__file__).resolve().parents[3] / 'data'/ 'abstract_2005_2022_full.csv',
           'only_classifier_params': False,
           'disable_discriminator': False,
           'disable_nsp_loss': False,
           'disable_clm_loss': False,
           'varied_pairs' : False,
           ##### GPT configuration #####
        #    'checkpoint_path': '/home/liel-blu/project/saved_models/GAN_experiments_models/gpt2-medium_231116_101738.263843/epoch_14', #None if training from scratch
            'checkpoint_path': None,
           'base_model': 'gpt2-medium',
          #  'base_model': 'microsoft/biogpt',
          #  'base_model': 'all-MiniLM-L6-v2',
           # CLMLoss is only available for GPT-based models!!! 
           'loss' : 'CLMLoss',
           ##### BERT configuration #####
           'sts_pairs': True,
           'sbert_loss_margin': 0.5,
           'batch_size': 8,
           'test_size': 0.3,
           'bert_pretrained_over_pubMed_path': Path(__file__).resolve().parents[4] / 'saved_models' / 'bert_tiny_uncased_2010_2018_v2020_epoch39',
           'bert_tokenizer': os.path.join('google', 'bert_uncased_L-2_H-128_A-2'),
           'similarity_metric': 'cross_entropy',
           'cosine_similarity_clean_method': '',
           'reset_different_topic_entries_flag': 1,
           'max_length_bert_input': 50,
           'max_sentences_per_abstract': 20,
           'SAVE_PATH': Path(__file__).resolve().parents[4] / 'saved_models' /'GAN_experiments_models' ,
           }
