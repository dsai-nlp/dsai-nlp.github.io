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

Apply the tokenizer to all paragraphs in the training and validation datasets. Convert all words into lowercase.

**Sanity check**: after this step, your training set should consist of around 147,000 paragraphs and the validation set around 18,000 paragraphs. (The exact number depends on what tokenizer you selected.)

## Step 2: Encoding the text as integers

### Building the vocabulary

Create a utility (a function or a class) that goes through the training text and creates a *vocabulary*: a mapping from token strings to integers.

In addition, the vocabulary should contain 3 special symbols:
- a symbol for previously unseen or low-frequency tokens,
- a symbol we will put at the beginning of each paragraph,
- a symbol we will put at the end of each paragraph.

The total size of the vocabulary (including the 3 symbols) should be at most `max_voc_size`, which is is a user-specified hyperparameter. If the number of unique tokens in the text is greater than `max_voc_size`, then use the most frequent ones.

<details>
<summary>A `nn.Counter` can be convenient when computing the frequencies.</summary.>

x

y

z
</details>

**Sanity check**: after creating the vocabulary, make sure that
- the size of your vocabulary is not greater than the max vocabulary size you specified,
- the 3 special symbols exist in the vocabulary and that they don't coincide with any real words,
- that some highly frequent example words (e.g. "the", "and") are included in the vocabulary but that some rare words (e.g. "cuboidal", "epiglottis") are not.

### Encoding the texts and creating training instances

We will now collect training instances for our language model, where we learn to predict the next token given the previous *N* tokens.

Go through the training and validation data and extract all sequences of *N*+1 tokens and map them to the corresponding integer values. Remember to use the special symbols when necessary:
- the "unseen" symbol for tokens not in your vocabulary,
- *N* "beginning" symbols before each paragraph,
- an "end" symbol after each paragraph.

Store all these sequences in lists.

**Sanity check**: after these steps, you should have around 12 million training instances and 1.5 million validation instances.

## Step 3: Developing a language model

### Setting up the neural network structure

Set up a neural network inspired by the neural language model proposed by [Bengio et al. (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). The main components are:
- an *embedding layer* that maps token integers to floating-point vectors,
- *intermediate layers* that map between input and output representations,
- an *output layer* that computes (the logits of) a probability distribution over the vocabulary.

You are free to experiment with the design of the intermediate layers and you don't have to follow the exact structure used in the paper.

<details>
<summary> <b>Side note</b>: Setting up a neural network in PyTorch (click to expand)</summary>
There are a few different ways that we can write code to set up a neural network in PyTorch.

If your model has the traditional structure of stacked layers, then the most concise way to declare the model is to use `nn.Sequential`:

```
model = nn.Sequential(
  layer1,
  layer2,
  ...
  layerN)
```
You can use any type of layers here. In our case, you'll typically start with a `nn.Embedding` layer, followed by some intermediate layers (e.g. `nn.Linear` followed by some activation such as `nn.ReLU`), and then a linear output layer.

A more general solution is to declare your network as a class that inherits from `nn.Module`. You will then have to declare your model components in `__init__` and define the forward computation in `forward`:
```
class MyNetwork(nn.Module):
  def __init__(self, hyperparameters):
    super().__init__()
    self.layer1 = ... some layer ...
    self.layer2 = ... some layer ...
    ...

  def forward(self, inputs):
    step1 = self.layer1(inputs)
    step2 = self.layer2(step1)
    ...
    return something
``` 

The second coding style, while more verbose, has the advantage that it is easier to debug: for instance, it is easy to check the shapes of intermediate computations. It is also more flexible and allows you to go beyond the constraints of a traditional layered setup.
</details>

**Hint**: a ``nn.Flatten`` layer is a convenient tool that you can put after the embedding layer to get the right tensor shapes. Let's say we have a batch of *B* inputs, each of which is a context window of size *N*, so our input tensor has the shape (*B*, *N*). The output from the embedding layer will have the shape (*B*, *N*, *D*) where *D* is the embedding dimensionality. If you use a ``nn.Flatten``, we go back to a two-dimensional tensor of shape (*B*, *N* * *D*). That is, we can see this as a step that concatenates the embeddings of the tokens in the context window.

**Sanity check**: carry out the following steps:
- Create an integer tensor of shape 1x*N* where *N* is the size of the context window. It doesn't matter what the integers are except that they should be less than the vocabulary size. (Alternatively, take one instance from your training set.)
- Apply the model to this input tensor. It shouldn't crash here.
- Make sure that the shape of the returned output tensor is 1x*V* where *V* is the size of the vocabulary.

### Training the model

**Hint**: while developing the code, work with very small datasets. Monitor the cross-entropy loss (and/or the perplexity) over the training: if the loss does not decrease while you are training, there is probably an error.

### Evaluating



Compute the [perplexity](https://huggingface.co/docs/transformers/perplexity) of your model on the validation set.

**Hint**: the perplexity is `exp` applied to the mean of the negative log probability of each token. The cross-entropy loss can be practical here, since it computes the mean negative log probability.

## Step 4: Inspecting word embeddings

Testing.