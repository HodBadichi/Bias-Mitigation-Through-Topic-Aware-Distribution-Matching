hparams = {'learning_rate': 2e-5,
           'discriminator_factor': 0.5,
           'mlm_factor': 0.5,
           'gpus': 2,
           'max_epochs': 40,
           'gender_and_topic_path': '../../data/abstract_2005_2020_gender_and_topic.csv',
           'batch_size': 32,
           'test_size': 0.7,
           'bert_pretrained_over_pubMed_path': '../Models/bert_pretrain/bert_tiny_uncased_abstract_2005_2020_gender_and_topic_epochs/bert_tiny_uncased_abstract_2005_2020_gender_and_topic_39_',
           'bert_tokenizer': 'google/bert_uncased_L-2_H-128_A-2',
           'similarity_metric': 'cross_entropy',
           'reset_different_topic_entries_flag': 1,
           'SimilarityMatrixPath': '../NoahArc/CE_sim_matrix',
           'ProbabilityMatrixPath': '../NoahArc/CE_prob_matrix_reset_different_topic_entries_flag',
           'max_length_bert_input': 50,
           'max_sentences_per_abstract': 20,
           'SAVE_PATH': "/home/mor.filo/nlp_project/DistributionMatching/PubMed/Models"
           }