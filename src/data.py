# -*- coding: utf-8 -*-

import os
import torch
from collections import defaultdict

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

        self.unknown_token_string = '<unk>'

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def build_vocab(self, fname):
        # read a new vocabulary in from corpus
        with open(fname) as fh:
            tokens = 0
            for line in f:
                labels = line.split()
                tokens += len(words)
                for label in labels:
                    self.add_word(word)

    def unknown_token(self):
        return self.add_word(self.unknown_token_string)

    def __len__(self):
        return len(self.idx2word)

    def save(self, fh):
        print('\n'.join(self.idx2word), file=fh)

    def load(self, fname):
        with open(fname, encoding='utf-8') as fh:
            self.idx2word = [word.strip() for word in fh]
        self.word2idx = {word:idx for idx,word in enumerate(self.idx2word)}

class Corpus(object):
    def __init__(self, train_file, valid_file, test_file, vocab_file=None, permutation=None, test_length=None):
        self.dictionary = Dictionary()
        if vocab_file is not None:
            self.dictionary.load(vocab_file)
        else:
            self.build_vocab(train_file)
        self.train = self.tokenize(train_file, permutation=permutation)
        self.valid = self.tokenize(valid_file)
        self.test = self.tokenize(test_file, lines_to_keep=test_length)

    def build_vocab(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, path, permutation=None, lines_to_keep=None):
        """Tokenizes a text file."""
        if path is None:
            return None
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf-8') as fh:
            f = fh.readlines()
        if lines_to_keep is not None:
            # only keep first lines in file
            f = f[:lines_to_keep]
        if permutation is not None:
            f = [f[i] for i in permutation]

        words = []
        for line in f:
            words += line.split() + ['<eos>']

        # Tokenize file content
        ids = torch.LongTensor(len(words))
        for i, word in enumerate(words):
            if word in self.dictionary.word2idx:
                ids[i] = self.dictionary.word2idx[word]
            else:
                ids[i] = self.dictionary.unknown_token()

        return ids
