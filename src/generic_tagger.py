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
import model

parser = argparse.ArgumentParser(description='A generic tagging model, taking a corpus and its tags as input.')

# Model parameters.
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--vocab-file', type=str,
                    help='location of newline-separated list of all words sorted by index')
parser.add_argument('--train-file', type=str,
                    help='location of the train data corpus')
parser.add_argument('--valid-file', type=str,
                    help='location of the valid data corpus')
parser.add_argument('--test-file', type=str,
                    help='location of the test data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--test-length', type=int, default=None,
                    help='number of lines to test in the test corpus')
parser.add_argument('--tag-suffix', type=str, default='.tag')
parser.add_argument('--save-dir', type=str)
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
    help='report interval')
parser.add_argument('--save-model', type=str)

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

corpus = data.Corpus(args.train_file+'.tok', args.valid_file+'.tok', args.test_file+'.tok', vocab_file=args.vocab_file, test_length=args.test_length)
pos_corpus = data.Corpus(args.train_file+args.tag_suffix, args.valid_file+args.tag_suffix, args.test_file+args.tag_suffix, test_length=args.test_length)

model = model.RNNModel(args.model, len(corpus.dictionary), len(pos_corpus.dictionary), args.emsize, args.nhid, args.nlayers, args.dropout)

if args.cuda:
    model.cuda()
else:
    model.cpu()

model.requires_grad = False

criterion = nn.CrossEntropyLoss()

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

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

print('Batching eval text data.')
train_data = batchify(corpus.train, args.batch_size)
valid_data = batchify(corpus.valid, args.batch_size)
test_data = batchify(corpus.test, args.batch_size)

print('Batching eval pos data.')
train_pos_tags = batchify(pos_corpus.train, args.batch_size)
valid_pos_tags = batchify(pos_corpus.valid, args.batch_size)
test_pos_tags = batchify(pos_corpus.test, args.batch_size)

def get_pos_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    return Variable(source[i+1:i+1+seq_len].view(-1))

idx2word = corpus.dictionary.idx2word
word2idx = corpus.dictionary.word2idx
vocab_size = len(corpus.dictionary.idx2word)

def evaluate(data_source, pos_tags_source):
    ntokens = len(pos_corpus.dictionary)
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    num_batches = len(data_source) // args.bptt
    total_loss = 0
    for batch,i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
        data, _ = get_batch(data_source, i, evaluation=True)
        targets = get_pos_batch(pos_tags_source, i)

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += criterion(output_flat, targets)

        hidden = repackage_hidden(hidden)

    elapsed = time.time() - start_time
    print('| {:5d} batches | ms/batch {:5.2f} |'.format(
            num_batches, elapsed * 1000 / num_batches))

    return total_loss.data[0] / num_batches

def train(epoch, lr):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(pos_corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, _ = get_batch(train_data, i, evaluation=False)
        targets = get_pos_batch(train_pos_tags, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# At any point you can hit Ctrl + C to break out of training early.
lr = args.lr
best_val_loss = None
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch, lr)
        val_loss = evaluate(valid_data, valid_pos_tags)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if best_val_loss is None or val_loss < best_val_loss:
            with open(args.save_model, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save_model, 'rb') as f:
    model = torch.load(f)

# Run on test data.
print("test data size ", test_data.size())
constructor = hooks.NetworkSubspaceConstructor(model, args.save_dir)
constructor.add_hooks_to_model()
test_loss = evaluate(test_data, test_pos_tags)
constructor.remove_hooks()

print('=' * 89)
print('| End of training | test loss {:5.2f} '.format(
    test_loss))
print('=' * 89)
