import nltk.data
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import wordnet
import re
import json


class TextUtils:
    def __init__(self):
        nltk.download('punkt')
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def split_abstract_to_sentences(self, abstract):
        parts = abstract.split(';')
        sentences = []
        for part in parts:
            sentences.extend(self.sent_tokenizer.tokenize(part))
        return sentences

    @staticmethod
    def flatten_list_of_lists(lst):
        return [item for sublist in lst for item in sublist]

    def word_tokenize_abstract(self, abstract):
        sentences = self.split_abstract_to_sentences(abstract)
        words = []
        for s in sentences:
            words.extend(word_tokenize(s))
        return words

    @staticmethod
    def decide_on_word(word):
        w = word.lower().strip(string.punctuation)
        if TextUtils.represents_number(w):
            return "<NUMBER>"
        return w.lower()

    @staticmethod
    def represents_number(word_to_check):
        possible_numbers = re.findall(r'[\d\.]+', word_to_check)
        if len(possible_numbers) > 0 and len(possible_numbers[0]) == len(word_to_check):
            return True
        return False

    @staticmethod
    def filter_word_list(word_list):
        # decide_on_word removes punctuation, lowercases the words and replaces numbers with a specific token.
        return [w for w in map(TextUtils.decide_on_word, word_list) if len(w) > 0]

    def word_tokenize(self, text):
        return TextUtils.filter_word_list(self.word_tokenize_abstract(text))


def should_keep_sentence(sentence):
    blacklist = ['http', 'https', 'url', 'www', 'clinicaltrials.gov', 'copyright', 'funded by', 'published by',
                 'subsidiary', '©', 'all rights reserved']
    s = sentence.lower()
    for w in blacklist:
        if w in s:
            return False
    # re, find NCTs
    if len(re.findall('nct[0-9]+', s)) > 0:
        return False
    if len(sentence) < 40:
        return False
    return True


def clean_abstracts(df, abstract_field='title_and_abstract', output_sentences_field='broken_abstracts'):
    text_utils = TextUtils()
    # filter sentences
    d = {'total': 0, 'remaining': 0}

    def filter_sentences_and_words(abstract):
        sentences = text_utils.split_abstract_to_sentences(abstract)
        new_sentences = [sent for sent in sentences if should_keep_sentence(sent)]
        d['total'] += len(sentences)
        d['remaining'] += len(new_sentences)
        sent_list_filtered_by_words = [' '.join(text_utils.word_tokenize(sent)) for sent in new_sentences]
        return '<BREAK>'.join(sent_list_filtered_by_words)

    df[output_sentences_field] = df[abstract_field].apply(filter_sentences_and_words)
    df = df.dropna(subset=['broken_abstracts'], axis=0)
    print(f"kept {d['remaining']}/{d['total']} sentences")
    return df


def break_sentence_batch(samples):
    indexes = []
    all_sentences = []
    index = 0
    max_len = 0
    for sample in samples:
        sample_as_list = sample.split('<BREAK>')
        # sample is a list of sentences
        indexes.append((index, index + len(sample_as_list)))
        index += len(sample_as_list)
        all_sentences.extend(sample_as_list)
        if max_len < len(sample_as_list):
            max_len = len(sample_as_list)
    return indexes, all_sentences, max_len