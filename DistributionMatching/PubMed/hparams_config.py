import os


hparams = {'learning_rate': 2e-5,
           'discriminator_factor': 0.5,
           'mlm_factor': 0.5,
           'gpus': 1,
           'max_epochs': 40,
           'gender_and_topic_path': os.path.join(os.pardir,os.pardir,'data','abstract_2005_2020_gender_and_topic.csv'),
           'batch_size': 16,
           'test_size': 0.3,
           'bert_pretrained_over_pubMed_path': os.path.join(os.pardir,
                                                            'Models',
                                                            'bert_pretrain',
                                                            'bert_tiny_uncased_abstract_2005_2020_gender_and_topic_epochs',
                                                            'bert_tiny_uncased_abstract_2005_2020_gender_and_topic_39_'),
           'bert_tokenizer': os.path.join('google','bert_uncased_L-2_H-128_A-2'),
           'similarity_metric': 'cross_entropy',
           'cosine_similarity_clean_method': '',
           'reset_different_topic_entries_flag': 1,
           'SimilarityMatrixPathTrain': 'CE_sim_matrix_train',
           'ProbabilityMatrixPathTrain': 'CE_prob_matrix_reset_different_topic_entries_flag_train',
           'SimilarityMatrixPathVal': 'CE_sim_matrix_val',
           'ProbabilityMatrixPathVal': 'CE_prob_matrix_reset_different_topic_entries_flag_val',
           'SimilarityMatrixPathTest': 'CE_sim_matrix_test',
           'ProbabilityMatrixPathTest': 'CE_prob_matrix_reset_different_topic_entries_flag_test',
           'max_length_bert_input': 50,
           'max_sentences_per_abstract': 20,
           'SAVE_PATH': "/home/mor.filo/nlp_project/DistributionMatching/PubMed/Models",
           'DEBUG_FLAG': False
           }

# 'SimilarityMatrixPath': '../NoahArc/CE_sim_matrix',
# 'ProbabilityMatrixPath': '../NoahArc/CE_prob_matrix_reset_different_topic_entries_flag',