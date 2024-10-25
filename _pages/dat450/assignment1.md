---
layout: page
title: 'DAT450/DIT247: Programming Assignment 1'
permalink: /courses/dat450/assignment1/
description:
nav: false
nav_order: 4
---

# DAT450/DIT247: Programming Assignment 1

Our goal in this assignment is to implement a neural network-based language model similar to the one described by [Bengio et al. (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).

## Step 0: Preliminaries

Download the following texts.

## Step 1: Preprocessing the text

You will need a *tokenizer* that splits English text into separate words. In this assignment, you will just use an existing tokenizer. Popular NLP libraries such as SpaCy and NLTK come with built-in tokenizers. We recommend NLTK in this assignment since it is somewhat faster than SpaCy and somewhat easier to use.



## Step 2: Encoding the text as integers

### Building the vocabulary



**Sanity check**: make sure that
- the size of your vocabulary is not greater than the max vocabulary size you specified,
- the 3 special symbols exist in the vocabulary
- that some highly frequent English word (e.g. "the") is mapped to a small integer

### Encoding the texts

## Step 3: Developing a language model

### Splitting the text into *n*-grams

### Setting up the neural network structure

**Sanity check**: 

### Training the model

**Hint**: while developing the code, work with very small datasets. Monitor the cross-entropy loss (and/or the perplexity) over the training: if the loss does not decrease while you are training, there is probably an error.

### Evaluating

Compute the [perplexity](https://huggingface.co/docs/transformers/perplexity) of your model on the validation set.

**Hint**: the perplexity is `exp` applied to the mean of the negative log probability of each token. The cross-entropy loss can be practical here, since it computes the mean negative log probability.

## Step 4: Inspecting word embeddings

Testing.