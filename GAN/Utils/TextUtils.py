import string
import re

import nltk.data
from nltk.tokenize import word_tokenize


class TextUtils:
    def __init__(self):
        nltk.download('punkt')
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def SplitAbstractToSentences(self, abstract):
        parts = abstract.split(';')
        sentences = []
        for part in parts:
            sentences.extend(self.sent_tokenizer.tokenize(part))
        return sentences

    def WordTokenizeAbstract(self, abstract):
        sentences = self.SplitAbstractToSentences(abstract)
        words = []
        for s in sentences:
            words.extend(word_tokenize(s))
        return words

    @staticmethod
    def DecideOnWord(word):
        w = word.lower().strip(string.punctuation)
        if TextUtils.RepresentsNumber(w):
            return "<NUMBER>"
        return w.lower()

    @staticmethod
    def RepresentsNumber(word_to_check):
        possible_numbers = re.findall(r'[\d\.]+', word_to_check)
        if len(possible_numbers) > 0 and len(possible_numbers[0]) == len(word_to_check):
            return True
        return False

    @staticmethod
    def FilterWordList(word_list):
        # DecideOnWord removes punctuation, lowercases the words and replaces numbers with a specific token.
        return [w for w in map(TextUtils.DecideOnWord, word_list) if len(w) > 0]

    def WordTokenize(self, text):
        return TextUtils.FilterWordList(self.WordTokenizeAbstract(text))


def ShouldKeepSentence(sentence):
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


def CleanAbstracts(df, abstract_field='title_and_abstract', output_sentences_field='broken_abstracts'):
    text_utils = TextUtils()
    # filter sentences
    d = {'total': 0, 'remaining': 0}

    def FilterSentencesAndWords(abstract):
        sentences = text_utils.SplitAbstractToSentences(abstract)
        new_sentences = [sent for sent in sentences if ShouldKeepSentence(sent)]
        d['total'] += len(sentences)
        d['remaining'] += len(new_sentences)
        sent_list_filtered_by_words = [' '.join(text_utils.WordTokenize(sent)) for sent in new_sentences]
        return '<BREAK>'.join(sent_list_filtered_by_words)

    df[output_sentences_field] = df[abstract_field].apply(FilterSentencesAndWords)
    df = df.dropna(subset=['broken_abstracts'], axis=0)
    print(f"kept {d['remaining']}/{d['total']} sentences")
    return df


def BreakSentenceBatch(samples):
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
