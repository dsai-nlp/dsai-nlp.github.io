---
layout: page
title: 'DAT450/DIT247: Programming Assignment 2: Generating text from a language model'
permalink: /courses/dat450/assignment2/
description:
nav: false
nav_order: 4
---

# DAT450/DIT247: Programming Assignment 2: Generating text from a language model

In this assignment, we extend the models we investigated in the previous assignment in two different ways:
- In the previous assignment, we used a model that takes a fixed number of previous words into account. Now, we will use a model capable of considering a variable number of previous words: a *recurrent neural network*. (Optionally, you can also investigate *Transformers*.)
- In this assignment, we will also use our language model to generate texts.

### Pedagogical purposes of this assignment
- Investigating more capable neural network architectures for language modeling.
- Understanding text-generating algorithms.

### Requirements

Please submit your solution in [Canvas](https://chalmers.instructure.com/courses/31739/assignments/98455). **Submission deadline**: November 18.

Submit a notebook containing your solution to the programming tasks described below. This is a pure programming assignment and you do not have to write a technical report or explain details of your solution in the notebook: there will be a separate individual assignment where you will answer some conceptual questions about what you have been doing here.

## Step 0: Preliminaries

Make sure you have access to your solution for Programming Assignment 1 since you will reuse some parts.

Copy the tokenization and integer encoding part into a new notebook.

## Step 1: Adapting your code for RNNs

### Adapting the preprocessing

In the previous assignment, you developed preprocessing tools that extracted fixed-length sequences from the training data. You will now adapt the preprocessing so that you can deal with inputs of variable length.

**Splitting**: While we will deal with longer sequences than in the previous assignment, we'll still have to control the maximal sequence length (or we'll run out of GPU memory). Define a hyperparameter `max_sequence_length` and split your equences into pieces that are at most of that length. (Side note: in RNN training, limiting the sequence length is called <a href="https://d2l.ai/chapter_recurrent-neural-networks/bptt.html"><em>truncated backpropagation through time</em></a>.)

**Padding**: In the previous assignment, you developed a tool that finds the most frequent words in order to build a vocabulary. In this vocabulary, you defined special symbols to cover a number of corner cases: the beginning and end of text passages, and when a word is previously unseen or too infrequent.
Now, change your vocabulary builder to include a new special symbol that we will call *padding*: this will be used when our batches contain texts of different lengths.

After these changes, preprocess the text and build the vocabulary as in the previous assignment. Store the integer-encoded paragraphs in two lists, corresponding to the training and validation sets. 

**Sanity check**: You should have around 147,000 training paragraphs and 18,000 validation paragraphs. However, since you split the sequences, you will in the end get a larger number of training and validation instances. (The exact numbers depend on `max_sequence_length`.

### Adapting the batcher

In the previous assignment, you implemented some function to create training batches: that is, to put some number of training instances into a PyTorch tensor.

Now, change your batching function so that it can deal with sequences of variable lengths.
Since the output of the batching function are rectangular tensors, you need to *pad* sequences so they are of the same length.
So for each instance that is shorter than the longest instance in the batch, you should append the padding symbol until it has the right length.

**Sanity check**: Inspect a few batches. Make sure that they are 2-dimensional integer tensors with *B* rows, where *B* is the batch size you defined. The number of columns probably varies from batch to batch, but should never be longer than `max_sequence_length` you defined previously.
The integer-encoded padding symbol should only occur at the end of sequences.

## Step 2: Designing a language model using a recurrent neural network

### Setting up the neural network structure

Define a neural network that implements an RNN-based language model. It should include the following layers:

- an *embedding layer* that maps token integers to floating-point vectors,
- an *recurrent layer* implementing some RNN variant (we suggest [`nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) or [`nn.GRU`](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)),
- an *output layer* that computes (the logits of) a probability distribution over the vocabulary.

You will have to define some hyperparameters such as the embedding size (as in the previous assignment) and the size of the RNN's hidden state.

<details>
<summary><b>Hint</b>: If you are doing the batching as recommended above, you should set <code>batch_first=True</code> when declaring the RNN.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
The input to an RNN is a 3-dimensional tensor. If we set <code>batch_first=True</code>, then we assume that the input tensor is arranged as (<em>B</em>, <em>N</em>, <em>E</em>) where <em>B</em> is the batch size, <em>N</em> is the sequence length, and <em>E</em> the embedding dimensionality. In this case, the RNN "walks" along the second dimension: that is, over the sequence of tokens.

If on the other hand you set <code>batch_first=False</code>, then the RNN walks along the first dimension of the input tensor and it is assumed to be arranged as (<em>N</em>, <em>B</em>, <em>E</em>).
</div>
</details>

<details>
<summary><b>Hint</b>: How to apply RNNs in PyTorch.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">

Take a look at the documentation of one of the RNN types in PyTorch. For instance, here is the documentation of <a href="https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"><code>nn.LSTM</code></a>. In particular, look at the section called <b>Outputs</b>. It is important to note here that all types of RNNs return <b>two</b> outputs when you call them in the forward pass. In this assignment, you will need the <b>first</b> of these outputs, which correspond to the RNN's output for each <em>token</em>. (The other outputs are the <em>layer-wise</em> outputs.)

As we discussed in the previous assignment, PyTorch allows users to set up neural networks in different ways: the more compact approach using <code>nn.Sequential</code>, and the more powerful approach by inheriting from <code>nn.Module</code>.

If you implement your language model by inheriting from <code>nn.Module</code>, just remember that the RNN gives two outputs in the forward pass, and that you just need the first of them.

<pre>
class MyRNNBasedLanguageModel(nn.Module):
  def __init__(self, ... ):
    super().__init__()
    ... initialize model components here ...
    
  def forward(self, batch):
    embedded = ... apply the embedding layer ...
    rnn_out, _ = self.rnn(embedded)
    ... do the rest ...
</pre>

If you define your model using a <code>nn.Sequential</code>, we need a workaround to deal with the complication that the RNN returns two outputs. Here is one way to do it.
<pre>
class RNNOutputExtractor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, rnn_out):
        return rnn_out[0]
</pre>
The <code>RNNOutputExtractor</code> can then be put after the RNN in your list of layers.
</div>
</details>

**Sanity check**: carry out the following steps:
- Create an integer tensor of shape 1x*N* where *N* is the length of the sequene. It doesn't matter what the integers are except that they should be less than the vocabulary size. (Alternatively, take one instance from your training set.)
- Apply the model to this input tensor. It shouldn't crash here.
- Make sure that the shape of the returned output tensor is 1x*N*x*V* where *V* is the size of the vocabulary. This output corresponds to the logits of the next-token probability distribution, but it is useless at this point because we haven't yet trained the model.

### Training the model

Adapt your training loop from the previous assignment, with the following changes

<details>
<summary><b>Hint</b>: the output tensor is the input tensor, shifted one step to the right.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
For instance, let's say our training text is <em>This is great !</em> (in practice, the words will be integer-coded).
That means that at the first word (<em>This</em>), we want the model to predict the second word (<em>is</em>). At the second word, the goal is to predict <em>great</em>, and so on.

So when you process a batch in the training loop, you should probably split it into an input and an output part:
<pre>
input_tokens = batch[:, :-1]
output_tokens = batch[:, 1:]
</pre>
</div>
This means that the input consists of all the columns in the batch except the last one, and the output of all the columns except the first one.
</details>

<details>
<summary><b>Hint</b>: how to apply the loss function when training a language model.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
The loss function (<a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"><code>CrossEntropyLoss</code></a>) expects two input tensors:
<ul>
<li>the <em>logits</em> (that is: the unnormalized log probabilities) of the predictions,</li>
<li>the <em>targets</em>, that is the true output values we want the model to predict.</li>
</ul>

Here, the tensor is expected to be one-dimensional (of length <em>B</em>, where <em>B</em> is the batch size) and the logits tensor to be two-dimensional (of shape (<em>B</em>, <em>V</em>) where <em>V</em> is the number of choices).

In our case, the loss function's expected input format requires a small trick, since our targets tensor is two-dimensional (<em>B</em>, <em>N</em>) where <em>N</em> is the maximal text length in the batch. Analogously, the logits tensor is three-dimensional (<em>B</em>, <em>N</em>, <em>V</em>). To deal with this, you need to reshape the tensors before applying the loss function.
<pre>
targets = targets.view(-1)                  # 2-dimensional -> 1-dimensional
logits = logits.view(-1, logits.shape[-1])  # 3-dimensional -> 2-dimensional
</pre>
</div>
</details>

<details>
<summary><b>Hint</b>: take padding into account when defining the loss.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
When the loss is computed, we don't want to include the positions where we have inserted the dummy padding tokens.
<a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"><code>CrossEntropyLoss</code></a> has a parameter <code>ignore_index</code> that you can set to the integer you use to represent the padding tokens.
</div>
</details>

Run the training function and compute the perplexity on the validation set as in the previous assignment.

## Step 3: Generating text

### Predicting the next word

As a starting point, we'll repeat the exercise from the first assignment where we take some example

- Apply the model to the integer-encoded input text.
- Take the model's output at the last position.
- Use <a href="https://pytorch.org/docs/stable/generated/torch.argmax.html"><code>argmax</code></a> to find the index of the highest-scoring item.
- Apply the inverse vocabulary encoder (that you created in Step 2) so that you can understand what words the model thinks are the most likely in this context.

### Generating texts

Implement a random sampling algorithm as described in the recording ([video](https://youtu.be/QtwpM-OGOew), [pdf](http://www.cse.chalmers.se/~richajo/dat450/lectures/l4/m4_3.pdf)). The function should take the following inputs:

- `prompt`: the prompt that initializes the text generation.
- `max_length`: the maximal number of steps before terminating.
- `temperature`: controls the degree of randomness by scaling the predicted logits.
- `topk`: to implement top-K sampling, i.e. the next-word distribution is truncated so that it only includes the `topk` most probable tokens.

The text generation should proceed until it an end-of-text symbol has been generated, or for at most `max_length` steps.

<details>
<summary><b>Hint</b>: The <a href="https://pytorch.org/docs/stable/generated/torch.topk.html"><code>topk</code></a> function will be useful here.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
This function takes a tensor as input and returns the <em>k</em> highest scores and their corresponding indices.
</div>
</details>

**Sanity check**: There are two ways to make this random sampling algorithm behave like *greedy decoding* (that is: there is no randomness, and the most likely next word is selected in each step). Run the function in these two ways and make sure you get the same output in both cases.

**Optional tasks**: 
