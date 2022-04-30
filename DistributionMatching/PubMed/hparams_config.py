hparams = {'learning_rate': 5e5,
           'discriminator_factor': 0.5,
           'mlm_factor': 0.5,
           'gpus': 1,
           'max_epochs': 40,
           'gender_and_topic_path': '../../data/abstract_2005_2020_gender_and_topic.csv',
           'batch_size': 64,
           'test_size': 0.7,
           'bert_pretrained_over_pubMed_path': '',
           'bert_tokenizer': 'google/bert_uncased_L-2_H-128_A-2'
           }