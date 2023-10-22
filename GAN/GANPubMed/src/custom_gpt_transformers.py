from transformers import BioGptTokenizer, BioGptForCausalLM
from torch import nn
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer,  models, losses
from transformers import GPT2Tokenizer, GPT2LMHeadModel

"""
Custon Transformer models for Sbert in GPT embeddings that contain a CLM head for CLM loss calculation
"""

class BaseGPTTransformer(models.Transformer):
    def __init__(self, auto_model, tokenizer, max_seq_length: Optional[int] = None,
            model_args: Dict = {}, cache_dir: Optional[str] = None,
                do_lower_case: bool = False):
        nn.Module.__init__(self)
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case
        self.auto_model = auto_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__  


    #overriding forward method to output embeddings to be used by discriminator in GAN
    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False, output_embeddings=True)
        output_tokens = output_states
        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features  

class BioGPTTransformer(BaseGPTTransformer):
    def __init__(self,  max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                 do_lower_case: bool = False):
        auto_model = BioGptForCausalLM.from_pretrained('microsoft/biogpt')
        tokenizer = BioGptTokenizer.from_pretrained('microsoft/biogpt')
        super().__init__(auto_model, tokenizer, max_seq_length, model_args, cache_dir, do_lower_case)

class GPT2MediumTransformer(BaseGPTTransformer):
    def __init__(self, max_seq_length: Optional[int] = None, model_args: Dict = {}, cache_dir: Optional[str] = None, do_lower_case: bool = False):
        auto_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        super().__init__(auto_model, tokenizer, max_seq_length, model_args, cache_dir, do_lower_case)