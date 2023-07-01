import os
from pathlib import Path

config = {'PubMedData': Path(__file__).resolve().parents[3] / 'data'/ 'abstract_2005_2020_full.csv',
          'topic_model_path':  Path(__file__).resolve().parents[4] / 'saved_models' / 'bert_tiny_uncased_2010_2018_v2020_epoch39' / "BERTopicModel"
          }
