#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch
from torchtext import data
from datasets_helper import MR
from cnn import Text_CNN
from train import train
from gensim.models import KeyedVectors
import numpy as np


def load_MR(text_field, label_field, batch_size, seed, **kwargs):
    train_data, dev_data = MR.splits(text_field, label_field, seed=seed)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, len(dev_data)),
        **kwargs
    )
    return train_iter, dev_iter


def read_embeddings(embedding_file, vocab, dim, seed):
    print("Reading in wordvectors ...", end=' ')
    model = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    print("Done")

    np.random.seed(seed)
    emb = np.random.rand(len(vocab), dim)

    for i, w in enumerate(vocab.itos):
        try:
            vec = model.word_vec(w)
        except KeyError:
            continue

        emb[i, :] = vec

    return emb


def Main(params, learn_emb, pretrained,
         log_interval, test_interval, use_gpu, seed):
    print("Loading movie review data ...", end=' ')
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter = load_MR(
        text_field, label_field, params['batch_size'],
        seed,
        device=-1,
        repeat=False
    )
    print("Done")

    if pretrained is None:
        embeddings = None
    else:
        embeddings = read_embeddings(
            pretrained, text_field.vocab, params['embedding_dim'], seed
        )

    if use_gpu and not torch.cuda.is_available():
        print("CUDA is not available on your system. I will use CPU.")
        use_gpu = False

    print("\nParameters:")
    for k, v in sorted(params.iteritems()):
        print("\t{}={}".format(k.upper(), v))

    model = Text_CNN(
        len(text_field.vocab),
        params['embedding_dim'],
        2,
        params['num_kernels'],
        params['kernel_sizes'],
        seed,
        dropout=params['dropout'],
        learnable_embeddings=learn_emb,
        pretrained_embeddings=embeddings
    )

    # try replacing Adam with SGD
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params['learning_rate']
    )

    print("Start Training!")
    train(
        train_iter, dev_iter, model, params['num_epochs'], optimizer,
        use_cuda=use_gpu,
        log_interval=log_interval, test_interval=test_interval
    )


if __name__ == '__main__':
    seed = 5

    hyperparams = {
        'embedding_dim': 300,  # 128
        'num_kernels': 100,
        'kernel_sizes': [3, 4, 5],
        'learning_rate': 0.001,
        'num_epochs': 1,
        'batch_size': 50,
        'dropout': 0.5
    }

    learnable_embeddings = False  # True
    log_interval = 10  # batches
    test_interval = 10  # epochs
    use_gpu = False  # True

    pretrained = None
    # pretrained = "GoogleNews-vectors-negative300.bin.gz"

    Main(
        hyperparams,
        learnable_embeddings,
        pretrained,
        log_interval,
        test_interval,
        use_gpu,
        seed
    )
