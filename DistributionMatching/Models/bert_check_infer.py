import sys
sys.path.append('/home/mor.filo/nlp_project/')
import pandas as pd
from GAN.PubMed.text_utils import break_sentence_batch
from transformers import AutoTokenizer, BertForMaskedLM
import torch

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    documents_df = pd.read_csv(rf'../../data/abstract_2005_2020_gender_and_topic.csv', encoding='utf8')
    broken_abstracts = documents_df.head(5)['broken_abstracts']
    tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    bert_model = BertForMaskedLM.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    max_sentences = 20
    max_length = 50
    sentence_embedding_size = bert_model.get_input_embeddings().embedding_dim
    indexes, all_sentences, max_len = break_sentence_batch(broken_abstracts)
    inputs = tokenizer(all_sentences, padding=True, truncation=True, max_length=max_length,
                            add_special_tokens=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = bert_model(**inputs, output_hidden_states=True).hidden_states[-1]
    sample_embedding = []
    for start, end in indexes:
        cur_doc_embedding = outputs[start:end]
        if len(cur_doc_embedding) > max_sentences:  # Too many sentences
            cur_doc_embedding = cur_doc_embedding[:max_sentences]
            sample_embedding.append(torch.flatten(cur_doc_embedding))
        else:  # Too few sentences - add padding
            padding = torch.zeros(max_sentences - len(cur_doc_embedding), sentence_embedding_size,
                                  device=device)
            sample_embedding.append(torch.flatten(torch.cat([cur_doc_embedding, padding], dim=0)))
    aggregated = torch.stack(sample_embedding)