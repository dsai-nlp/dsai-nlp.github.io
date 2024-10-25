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

Download the following text files. They consist of Wikipedia articles converted into raw text.

## Step 1: Preprocessing the text

You will need a *tokenizer* that splits English text into separate words. In this assignment, you will just use an existing tokenizer. Popular NLP libraries such as SpaCy and NLTK come with built-in tokenizers. We recommend NLTK in this assignment since it is somewhat faster than SpaCy and somewhat easier to use.



## Step 2: Encoding the text as integers

### Building the vocabulary

Create a utility (a function or a class) that goes through the training text and creates a *vocabulary*: a mapping from token strings to integers.

In addition, the vocabulary should contain 3 special symbols:
- a symbol for previously unseen or low-frequency tokens,
- a symbol we will put at the beginning of each paragraph,
- a symbol we will put at the end of each paragraph.

The total size of the vocabulary (including the 3 symbols) should be at most `max_voc_size`, which is is a user-specified hyperparameter. If the number of unique tokens in the text is greater than `max_voc_size`, then use the most frequent ones.

**Sanity check**: after creating the vocabulary, make sure that
- the size of your vocabulary is not greater than the max vocabulary size you specified,
- the 3 special symbols exist in the vocabulary and that they don't coincide with any real words,
- that some highly frequent example words (e.g. "the", "and") are included in the vocabulary but that some rare words (e.g. "cuboidal", "epiglottis") don't.

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