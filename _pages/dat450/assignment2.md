---
layout: page
title: 'DAT450/DIT247: Programming Assignment 2: Generating text from a language model'
permalink: /courses/dat450/assignment2/
description:
nav: false
nav_order: 4
---

# DAT450/DIT247: Programming Assignment 2: Generating text from a language model

We extend the models we investigated in the previous assignment.

### Pedagogical purposes of this assignment
- Investigating more capable neural network architectures for language modeling
- Understanding text-generating algorithms

### Requirements

Please submit your solution in [Canvas](https://chalmers.instructure.com/courses/31739/assignments/98455). **Submission deadline**: November 18.

Submit a notebook containing your solution to the programming tasks described below. This is a pure programming assignment and you do not have to write a technical report or explain details of your solution in the notebook: there will be a separate individual assignment where you will answer some conceptual questions about what you have been doing here.

## Step 0: Preliminaries

Make sure you have access to your solution for Programming Assignment 1 since you will reuse some parts.

Copy the tokenization and integer encoding part into a new notebook.

## Step 1: Adapting the vocabulary builder

In the previous assignment, you developed a tool that finds the most frequent words in order to build a vocabulary. In this vocabulary, you defined special symbols to cover a number of corner cases: the beginning and end of text passages, and when a word is previously unseen or too infrequent.

Now, adapt your vocabulary builder to include a new special symbol that we will call *padding*: this will be used when our batches contain full texts but these texts are of different lengths.

Preprocess the text and build the vocabulary as in the previous assignment.

Store the integer-encoded paragraphs in two lists, corresponding to the training and validation sets. They should *not* be split into fixed-length windows as in the previous assignment.

**Sanity check**: after these steps, you should have around 147,000 training instances and 18,000 validation instances.

## Step 2: Adapting the batcher

In the previous assignment, we created training and validation instances by extracting sequences of a fixed length.

Write a function that takes a list of integer-encoded lists as inputs, and puts them into a PyTorch tensor.
- Define a maximal length: this can be a large number (e.g. around 1000) and truncate paragraphs that are longer than the maximal length.
- Pad the sequences to the right. 

The function you created can be used as the `collate_fn` in a `DataLoader`.

**Sanity check**: Inspect a few batches. Make sure that they are 2-dimensional integer tensors with *B* rows, where *B* is the batch size you defined. The number of columns probably varies from batch to batch, but should not be longer than the maximal passage length you defined.

## Step 3: Designing a language model using a recurrent neural network

### Setting up the neural network structure

Define a neural network that implements an RNN-based language model. It should include the following layers:

- an *embedding layer* that maps token integers to floating-point vectors,
- an *recurrent layer* implementing some RNN variant (we suggest [`nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) or [`nn.GRU`](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)),
- an *output layer* that computes (the logits of) a probability distribution over the vocabulary.

You will have to define some hyperparameters such as the embedding size (as in the previous assignment) and the size of the RNN's hidden state.

<details>
<summary><b>Hint</b>: If you are doing the batching as recommended above, you should set <code>batch_first=True</code> when declaring the RNN.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
The input to an RNN is a 3-dimensional tensor. If we set <code>batch_first=True</code>, then we assume that the input tensor is arranged as (*B*, *N*, *E*) where *B* is the batch size, *N* is the sequence length, and *E* the embedding dimensionality. In this case, the RNN "walks" along the second dimension: that is, over the sequence of tokens.

If on the other hand you set <code>batch_first=False</code>, then the RNN walks along the first dimension of the input tensor and it is assumed to be arranged as (*N*, *B*, *E*).
</div>
</details>

<details>
<summary><b>Hint</b>: How to apply RNNs in PyTorch.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
As we discussed in the previous assignment, PyTorch allows users to set up neural networks in different ways: the more compact approach using <code>nn.Sequential</code>, and the more powerful approach by inheriting from <code>nn.Module</code>.

</div>
</details>

<details>
<summary><b>Hint</b>: You won't need a <code>Flatten</code> this time.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">xxx
</div>
</details>

**Sanity check**: carry out the following steps:
- Create an integer tensor of shape 1x*N* where *N* is the length of the sequene. It doesn't matter what the integers are except that they should be less than the vocabulary size. (Alternatively, take one instance from your training set.)
- Apply the model to this input tensor. It shouldn't crash here.
- Make sure that the shape of the returned output tensor is 1x*N*x*V* where *V* is the size of the vocabulary. This output corresponds to the logits of the next-token probability distribution, but it is useless at this point because we haven't yet trained the model.

### Training the model

Adapt your training loop from the previous assignment, with the following changes

<details>
<summary><b>Hint</b>: take padding into account when defining the loss.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"><code>CrossEntropyLoss</code></a>
</div>
</details>

<details>
<summary><b>Hint</b>: the output tensor is the input tensor, shifted one step to the right.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
XYZ
</div>
</details>

<details>
<summary><b>Hint</b>: how to apply the loss to the tensors.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
XYZ
</div>
</details>

Compute the perplexity on the validation set.

## Step 4: Generating text

### Predicting the next word

As a starting point, we'll repeat the exercise from the first assignment where we take some example

- Apply the model to the integer-encoded input text.
- Take the model's output at the last position.
- Use <a href="https://pytorch.org/docs/stable/generated/torch.argmax.html"><code>argmax</code></a> to find the index of the highest-scoring item.
- Apply the inverse vocabulary encoder (that you created in Step 2) so that you can understand what words the model thinks are the most likely in this context.

### Greedy decoding

Implement a greedy decoding algorithm as described in the recording.

This algorithm should select the highest-scoring output token at each step. Each generated token becomes a new input token in the next step.

Write a function that takes an input prompt and generates an output text for a given number of steps.

Consider the following examples...

### Random sampling

Implement a random sampling algorithm as described in the recording.

Write a function that takes an input prompt and generates an output text for a given number of steps. The function should use a *temperature* value that scales the output logits to control the randomness of the generation process. It should also use top-k sampling.


