#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from builtins import input


def get_most_frequent(text, size=10000):
    freq = {}
    for tok in word_tokenize(text):
        try:
            freq[tok] += 1
        except KeyError:
            freq[tok] = 1

    if len(freq) <= size:
        return freq.keys()
    else:
        return sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:size]


def generate_vocabulary(words):
    vocab = {}
    for i, word in enumerate(words):
        vocab[word] = i
    return vocab


def generate_windows(text, size=2):
    tokens = word_tokenize(text)
    windows = []
    last_words = []
    for tok in tokens:
        last_words.append(tok)

        if len(last_words) == size:
            windows.append(' '.join(last_words))
            last_words.pop(0)

    return windows


def cooccurrence_matrix(
        text, window_size=2,
        max_vocab_size=20000, same_word_zero=False,
        vectorizer=CountVectorizer):
    most_freq_words = get_most_frequent(text, size=max_vocab_size)
    vocab = generate_vocabulary(most_freq_words)

    windows = generate_windows(text, size=window_size)
    count_model = vectorizer(vocabulary=vocab)
    X = count_model.fit_transform(windows)
    Xc = (X.T * X)

    if same_word_zero:
        Xc.setdiag(0)

    return Xc.todense(), count_model.vocabulary_


def nearest_neighbors(word, vocabulary, embeddings, n=10):
    try:
        idx = vocabulary[word]
    except KeyError:
        return None

    vec = embeddings[idx].reshape(1, -1)
    sim = cosine_similarity(vec, embeddings)
    nn = sorted(
        vocabulary.keys(),
        key=lambda w: sim[0][vocabulary[w]],
        reverse=True
    )[:n]
    nn = map(lambda n: (n, sim[0][vocabulary[n]]), nn)

    return nn


def nearest_neighbor_loop(embeddings, vocabulary):
    while(True):
        try:
            word = input("Please enter a word: ")
        except EOFError:
            print("Goodbye.")
            break

        if not word:
            print("Goodbye.")
            break

        nn = nearest_neighbors(word, vocabulary, embeddings)
        if nn is None:
            print("I don't know the word!")
            continue

        for neighbor in nn:
            print(*neighbor)
