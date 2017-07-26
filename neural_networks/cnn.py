#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F


class Text_CNN(torch.nn.Module):
    '''
    Inspired by the model in https://arxiv.org/abs/1408.5882
    '''
    
    def __init__(self, vocab_size, embedding_dim, num_classes, num_kernels,
                 kernel_sizes, seed, dropout=0.0, learnable_embeddings=True,
                 pretrained_embeddings=None
    ):
        super(Text_CNN,self).__init__()

        torch.manual_seed(seed)

        V = vocab_size
        D = embedding_dim
        C = num_classes
        Ci = 1
        Co = num_kernels
        Ks = kernel_sizes

        # initialize an embedding layer
        self.embed = torch.nn.Embedding(V, D)

        # load pretrained embeddings if available
        if pretrained_embeddings is not None:
            self.embed.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )

        # should embeddings be fine-tuned during training?
        self.embed.weight.requires_grad = learnable_embeddings

        # convolution layers
        self.convs1 = torch.nn.ModuleList(
            [torch.nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        )

        # dropout component
        self.dropout = torch.nn.Dropout(p=dropout)

        # final fully connected layer (linear projection)
        self.fc = torch.nn.Linear(len(Ks)*Co, C)


    def forward(self, x):
        '''
        x      : (batch_size, max_sentence_length) =: (N, L)
        result : (batch_size, num_classes)
        '''

        x = self.embed(x) # (N,L,D)

        x = x.unsqueeze(1) # (N,Ci,L,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,L), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1) # (N,len(Ks)*Co)

        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc(x) # (N,C)

        return logit
