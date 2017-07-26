#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, num_epochs, optimizer, use_cuda=False,
          log_interval=1, test_interval=10):
    if use_cuda:
        model.cuda()

    steps = 0
    model.train()
    for epoch in range(1, num_epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if use_cuda:
                feature, target = feature.cuda(), target.cuda()

            # one training step #
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            # one training step #

            # logging
            steps += 1
            if steps % log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects) / batch.batch_size * 100.0
                sys.stdout.write(
                    '\rBatch[{:4}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
                        steps, 
                        loss.data[0], 
                        accuracy,
                        corrects,
                        batch.batch_size
                    )
                )

        # testing
        if epoch % test_interval == 0:
            print("\nEpoch {}".format(epoch))
            eval(dev_iter, model, use_cuda=use_cuda)

    print("\n\nTraining has finished!\n")

    print("Training Set")
    eval(train_iter, model, use_cuda=use_cuda)
    print("Dev Set")
    eval(dev_iter, model, use_cuda=use_cuda)


def eval(data_iter, model, use_cuda=False):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if use_cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = float(corrects) / size * 100.0
    model.train()
    print(
        'Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(
            avg_loss, 
            accuracy, 
            corrects, 
            size
        )
    )


def predict(text, model, text_field, label_field):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_field.vocab.itos[predicted.data[0][0]+1]
