# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:47:26 2018

@author: xiaodan.hu
"""

import tensorflow as tf
import numpy as np

# 训练集合为10000个
train_size = 50
seq_length = 10
test_size = 10
hidden_size = 10
learning_rate = 0.1
vocab_size = 2


Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
hprev = np.zeros((hidden_size, 1))  # reset RNN memory


inputs = np.random.randint(2, size=(train_size, seq_length, 1))
test_inputs = np.random.randint(2, size=(test_size, seq_length, 1))


def lossFun(input_, target_, hprev):

    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(len(input_)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][input_[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh,
                                                    hs[t - 1]) + bh)  # hidden state
        # unnormalized log probabilities for next chars
        ys[t] = np.dot(Why, hs[t]) + by
        # probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][target_[t], 0])  # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(
        Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(input_))):
        dy = np.copy(ps[t])
        # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dy[target_[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        # clip to mitigate exploding gradients
        np.clip(dparam, -1, 1, out=dparam)
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(input_) - 1]


for counter, input_ in enumerate(inputs):
    hprev = np.zeros((hidden_size, 1))  # reset RNN memory
    target_ = np.roll(input_,2)
    target_[0:2] = 0
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(input_, target_, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if counter % 100 == 0:
        print("current progress %d,loss %f" % (counter, smooth_loss))
    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / \
            np.sqrt(mem + 1e-8)  # adagrad update


for test_case in test_inputs:
   
    target_ = np.roll(test_case,2)
    target_[0:2]=0
    xs, hs, ys = {}, {}, {}
    h = np.zeros((hidden_size, 1))  # reset RNN memory
    output = []
    for t in range(len(test_case)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][test_case[t]] = 1
        h = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        if ix == 0:
            output.append(0)
        else:
            output.append(1)
            
    print("input")
    print(test_case.reshape(10).tolist())
    print("correct answer")
    print(target_.reshape(10).tolist())
    print("test answer")
    print(output)