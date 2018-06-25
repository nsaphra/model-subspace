# -*- coding: utf-8 -*-

import sys
import os
import numpy
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy import sparse
import pickle
import ast
import json

import data
import hooks

parser = argparse.ArgumentParser(description='Evaluate language model semantic and syntactic features')

# Model parameters.
parser.add_argument('--model-file', type=str,
                    help='model checkpoint to use')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--vocab-file', type=str,
                    help='location of newline-separated list of all words sorted by index')
parser.add_argument('--corpus-file', type=str,
                    help='location of the valid data corpus')
parser.add_argument('--original-src', type=str,
                    help='location of the modules required for the model')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--batch-size', type=int, default=60)
parser.add_argument('--test-length', type=int, default=None,
                    help='number of lines to test in the test corpus')
parser.add_argument('--save-dir', type=str)

args = parser.parse_args()

sys.path.append(args.original_src)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.model_file, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

model.requires_grad = False

corpus = data.Corpus(None, None, args.corpus_file, vocab_file=args.vocab_file, test_length=args.test_length)

def batchify(data, bsz, use_cuda=True):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda and use_cuda:
        data = data.cuda()
    return data

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=True)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

print('Batching eval text data.')
test_data = batchify(corpus.test, args.batch_size)

idx2word = corpus.dictionary.idx2word
word2idx = corpus.dictionary.word2idx
vocab_size = len(corpus.dictionary.idx2word)

criterion = nn.CrossEntropyLoss()

# Run on test data.
print("test data size ", test_data.size())

constructor = hooks.NetworkSubspaceConstructor(model, args.save_dir)

def evaluate(data_source):
    ntokens = len(corpus.dictionary)
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    num_batches = len(data_source) // args.bptt
    total_loss = 0
    for batch,i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
        data, targets = get_batch(data_source, i)

        constructor.set_word_sequence(targets)

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += criterion(output_flat, targets)

        hidden = repackage_hidden(hidden)

    elapsed = time.time() - start_time
    print('| {:5d} batches | ms/batch {:5.2f} |'.format(
            num_batches, elapsed * 1000 / num_batches))

    return total_loss.data[0] / num_batches

constructor.add_hooks_to_model()
test_loss = evaluate(test_data)
constructor.remove_hooks()

print('=' * 89)
print(test_loss)
print('=' * 89)
