---
layout: page
title: 'DAT450/DIT247: Programming Assignment 1: Introduction to language modeling'
permalink: /courses/dat450/assignment1/
description:
nav: false
nav_order: 4
---

# DAT450/DIT247: Programming Assignment 1: Introduction to language modeling

*Language modeling* is the foundation that recent advances in NLP technlogies build on. In essence, language modeling means that we learn how to imitate the language that we observe in the wild. More formally, we want to train a system that models the statistical distribution of natural language. Solving this task is exactly what the famous commercial large language models do (with some additional post-hoc tweaking to make the systems more interactive and avoid generating provocative outputs).

In the course, we will cover a variety of technical solutions to this fundamental task (in most cases, various types of Transformers). In this first assignment of the course, we are going to build a neural network-based language model that uses *recurrent* neural networks (RNNs) to model the interaction between words.

However, setting up the neural network itself is a small part of this assignment, and the main focus is on all the other steps we have to carry out in order to train a language model. That is: we need to process the text files, manage the vocabulary, run the training loop, and evaluate the trained models.

### About this document

The work for your submission is described in **Part 1&ndash;Part 4** below.

There are **Hints** at various places in the instructions. You can click on these **Hints** to expand them to get some additional advice.

### Pedagogical purposes of this assignment
- Introducing the task of language modeling,
- Getting experience of preprocessing text,
- Understanding the concept of word embeddings,
- Refreshing basic skills in how to set up and train a neural network,
- Introducing some parts of the HuggingFace ecosystem.

### Prerequisites

We expect that you can program in Python and that you have some knowledge of basic object-oriented programming. We will use terms such as "classes", "methods", "attributes", "functions" and so on.

On the theoretical side, you will need to remember fundamental concepts related to neural networks such as forward and backward passes, batches, initialization, optimization. 

On the practical side, you will need to understand the basics of PyTorch such as tensors, models, optimizers, loss functions and how to write the training loop. (If you need a refresher, there are plenty of tutorials available, for instance on the [PyTorch website](https://pytorch.org/tutorials/).) In particular, the [Optimizing Model Parameters tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) contains more or less everything you need to know for this assignment about PyTorch training loops.

### Submission requirements

Please submit your solution in [Canvas](https://canvas.chalmers.se/courses/36909/assignments/117614). 

**Submission deadline: November 10**.

Submit Python files containing your solution to the programming tasks described below.
In addition, to save time for the people who grade your submission, please submit a text file containing the outputs printed out by your Python program; read the instructions carefully so that the right outputs are included. (Most importantly: the perplexity evaluated on the validation set, and the next-word predictions.)

This is a pure programming assignment and you do not have to write a technical report or explain details of your solution: there will be a separate individual assignment where you will answer some conceptual questions about what you have been doing here.

## Part 0: Preliminaries

### Accessing the Minerva compute cluster

You can in principle solve this assignment on a regular laptop but it will be boring to train the full language model on a machine that does not have a GPU available. For this reason, we recommend to use the CSE department's compute cluster for education, called [Minerva](https://git.chalmers.se/karppa/minerva/-/blob/main/README.md). If you haven't used Minerva in previous courses, please read the instructions on the linked page.

In particular, read carefully the section called [**Python environments**](https://git.chalmers.se/karppa/minerva/-/blob/main/README.md#python-environments). For the assignments in the course, you can use an environment we have prepared for this course: `/data/courses/2025_dat450_dit247/venvs/dat450_venv`. (So to activate this environment, you type `source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate`.)

The directory `/data/courses/2025_dat450_dit247/assignments/a1` on Minerva contains two text files (`train.txt` and `val.txt`), which have been created from Wikipedia articles converted into raw text, with Wiki markup removed. In addition, there is a code skeleton (`A1_skeleton.py`) that contains stub implementations of the main pieces you need for your solution; for your own solution, you can copy this skeleton to your own directory.

### Suggested working approach with the cluster

Note that GPUs cannot be accessed from the JupyterHub notebooks, so you must submit SLURM jobs for your final deliverable.

<details>
<summary><b>Hint</b>: If you like to use VS Code, you have the option of connecting it to the cluster.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<ul>
<li>Install the <a href="https://code.visualstudio.com/docs/remote/ssh">Remote SSH extension</a>.</li>
<li>In the bottom left corner, you should have a small green button. Press this button. Alternatively, press Ctrl+Shift+P (Cmd+Shift+P on Mac) to open the command palette.</li>
<li>Select <tt>Connect to Host...</tt> or <tt>Remote SSH: Connect to Host...</tt></li>
<li>Type <code>YOUR_CID@minerva.cse.chalmers.se</code> and press enter. Enter your password if prompted.</li>
<li>Open your home folder from the menu File > Open folder. The home folder should be called <code>/data/users/YOUR_CID</code>.</li>
<li>If you want to use any extensions, they need to be installed separately on the VS Code server that is running on the cluster. Open the extension tab to install the extensions you need, e.g. the Python extension.</li>
</ul>
</div>
</details>

<details>
<summary><b>Hint</b>: While developing, you may optionally want to use interactive notebooks for a faster workflow. (But see the comment above about GPUs!)</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<ul>
<li>Read about <a href="https://git.chalmers.se/karppa/minerva/-/blob/main/README.md?ref_type=heads#jupyterhub">Minerva's JupyterHub</a></li>
<li>To make the course's Python environment available in notebooks, take the following steps:</li>
<ol>
<li>Log in on Minerva and activate the course environment.</li>
<li>Enter <code>python -m ipykernel install --user --name DAT450_venv --display-name "Python (DAT450_venv)"</code></li>
<li>If JupyterHub is running, restart it. Otherwise, start it now.</li>
<li>In the Launcher, you should now see an option called <code>Python (DAT450_venv)</code>.</li>
<li>If you create a notebook, you should be able to import libraries needed for the assignment, e.g. <code>import transformers</code></li>
</ol></li>
<li>If you keep your code in a Python file copied from <tt>A1_skeleton.py</tt>, then add the following somewhere in your notebook:
<pre>%load_ext autoreload
%autoreload 2
import your_a1_solution</pre>
By enabling auto-reloading, you won't have to restart the notebook every time you update the code in the Python file. Note that auto-reloading in notebooks does not work if you do <code>from your_a1_solution import ...</code>.

</li>
</div>
</details>


If you have questions about how to work with the cluster, please ask in the related [discussion thread](https://canvas.chalmers.se/courses/36909/discussion_topics/221739).

### Optional: Working on some other machine
If you are working on your own machine, make sure that the following libraries are installed:
- [NLTK](https://www.nltk.org/install.html) or [SpaCy](https://spacy.io/usage) for word splitting,
- [PyTorch](https://pytorch.org/get-started/locally/) for building and training the models,
- [Transformers](https://pytorch.org/get-started/locally/) and Datasets from HuggingFace,
- Optional: [Matplotlib](https://matplotlib.org/stable/users/getting_started/) and [scikit-learn](https://scikit-learn.org/stable/install.html) for the embedding visualization in the last step.
If you are using a Colab notebook, these libraries are already installed.

Then download and extract [this archive](https://www.cse.chalmers.se/~richajo/dat450/assignments/a1/a1.zip). It contains the text files and the code skeleton mentioned above.

## Part 1: Tokenization

**Terminological note**: It can be useful to keep in mind that people in NLP use the word ***tokenization*** in a couple of different ways. Traditionally, ***tokenization*** referred to the process of splitting texts into separate words. More recently, ***tokenization*** typically tends to mean all preprocessing steps we carry out to convert text into a numerical format suitable for neural networks. To avoid confusion, in this assignment we will use the term ***tokenization*** in the modern sense, and use the term ***word splitting*** otherwise.

### Using NLTK or SpaCy for word splitting

In this assignment, you will just use an existing library to split texts into words. Popular NLP libraries such as SpaCy and NLTK come with built-in functions for this purpose. We recommend NLTK in this assignment since it is somewhat faster than SpaCy and somewhat easier to use.

<details>
<summary><b>Hint</b>: How to use NLTK's English word splitter.</summary>

<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">Import the function <code>word_tokenize</code> from the <code>nltk</code> library. If you are running this on your own machine, you will first need to install NLTK with <code>pip</code> or <code>conda</code>. In Colab, NLTK is already installed.

For instance, <code>word_tokenize("Let's test!!")</code> should give the result <code>["Let", "'s", "test", "!", "!"]</code>
</div>
</details>

### Building the vocabulary

Each nonempty line in the text files correspond to one paragraph in Wikipedia. Apply the tokenizer to all paragraphs in the training and validation datasets. Convert all words into lowercase.

Create a function that goes through the training text and creates a *vocabulary*: a mapping from token strings to integers.

In addition, the vocabulary should contain 4 special symbols:
- a symbol for previously unseen or low-frequency tokens,
- a symbol we will put at the beginning of each paragraph,
- a symbol we will put at the end of each paragraph.
- a symbol we will use for *padding* so that we can make input tensors rectangular.

The total size of the vocabulary (including the 4 symbols) should be at most `max_voc_size`, which is is a user-specified hyperparameter. If the number of unique tokens in the text is greater than `max_voc_size`, then use the most frequent ones.

<details>
<summary><b>Hint</b>: A <a href="https://docs.python.org/3/library/collections.html#collections.Counter"><code>Counter</code></a> can be convenient when computing the frequencies.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">A <code>Counter</code> is like a regular Python dictionary, with some additional functionality for computing frequencies. For instance, you can go through each paragraph and call <a href="https://docs.python.org/3/library/collections.html#collections.Counter.update"><code>update</code></a>. After building the <code>Counter</code> on your dataset, <a href="https://docs.python.org/3/library/collections.html#collections.Counter.most_common"><code>most_common</code></a> gives the most frequent items.</div>
</details>

Also create some utility that allows you to go back from the integer to the original word token. This will only be used in the final part of the assignment, where we look at model outputs and word embedding neighbors.

**Example**: you might end up with something like this:
<pre>
str_to_int = { 'BEGINNING':0, 'END':1, 'UNKNOWN':2, 'PAD': 3, 'the':4, 'and':5, ... }

int_to_str = { 0:'BEGINNING', 1:'END', 2:'UNKNOWN', 3:'PAD', 4:'the', 5:'and', ... }
</pre>

**Sanity check**: after creating the vocabulary, make sure that
- the size of your vocabulary is not greater than the max vocabulary size you specified,
- the 4 special symbols exist in the vocabulary and that they don't coincide with any real words,
- some highly frequent example words (e.g. "the", "and") are included in the vocabulary but that some rare words (e.g. "cuboidal", "epiglottis") are not.
- if you take some test word, you can map it to an integer and then back to the original test word using the inverse mapping.

### Implementing a HuggingFace-like Tokenizer

Now, we turn to the task of implementing the utility that will turn a text into a numerical format that can be provided to neural networks as an input.
Our implementation will be functionally similar to the tokenizers provided by the **HuggingFace** library.

Write code for the missing parts in the `A1Tokenizer` in the skeleton Python file. You will need to implement the three methods `__init__`, `__call__`, and `__len__`. Most of the work will be done in `__call__`:  `__init__` is simply where you pass the information you need to set up the tokenize, and `__len__` should just return the size of the vocabulary.

<details>
<summary><b>Hint</b>: The weird-looking method <a href="https://docs.python.org/3/reference/datamodel.html#object"><code>__call__</code></a> is a special method that allows an object to be called like a function.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">That is: the following two lines are equivalent
<pre>tokenizer(some_texts)</pre>
and
<pre>tokenizer.__call__(some_texts)</pre>
</div>
</details>
&nbsp;

**Sanity check**: Apply your tokenizer to an input consisting of few texts and make sure that it seems to work. In particular, verify that the tokenizer can create a tensor output in a situation where the input texts do not contain the same number of words: in these cases, the shorter texts should be "padded" on the right side. For instance
```
tokenizer = (... create your tokenizer...)
test_texts = [['This is a test.', 'Another test.']]

tokenizer(test_texts, return_tensors='pt', padding=True,
          truncation=True)
```
The result should be something similar to the following example output (assuming that the integer 0 corresponds to the padding dummy token):
```
{'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 0, 0]]),
 'input_ids': tensor([[2, 35,  14,  11, 965,   6,  3],
                      [2, 153, 965,  6,   3,   0,  0]])}
```
Verify that at least the `input_ids` tensor corresponds to what you expect. (As mentioned in the skeleton code, the `attention_mask`</code>` is optional for this assignment.)

## Part 2: Loading the text files and creating batches

(This part just introduces some functionalities you may find useful when processing the data: it functions as a stepping stone for what you will do in Part 4. You do not have to include solutions to this part in your submission.)

**Loading the texts.** We will use the [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) library to load the texts from the training and validation text files. (You may feel that we are overdoing it, since these are simple text files, but once again we want to introduce you to the standard ecosystem used in NLP.)

```
from datasets import load_dataset
dataset = load_dataset('text', data_files={'train': TRAIN_FILE, 'val': VAL_FILE})
```

The training and validation sections can now be accessed as `dataset['train']` and `dataset['val']` respectively. The datasets internally use the [Arrow](https://arrow.apache.org/docs/index.html) format for efficiency; in practice, they can be accessed as if they were regular Python lists. That is: you can write `dataset['train'][8]` to access the 8th text in the training set.

Each instance in the training and validation sets correspond to Wikipedia paragraphs. 
Now, remove empty lines from the data:

```
dataset = dataset.filter(lambda x: x['text'].strip() != ''
```

**Sanity check**: after loading the datasets and removing empty lines, you should have around 147,000 training and 18,000 validation instances.

Optionally, it can be useful in the development phase to work with smaller datasets. The following is one way of achieving that:

```
from torch.utils.data import Subset
for sec in ['train', 'val']:
    dataset[sec] = Subset(dataset[sec], range(1000))
```

**Iterating through the datasets.**
When training and running neural networks, we typically use *batching*: that is, to improve computational efficiency, we process several instances in parallel.
We will use the `DataLoader` utility from PyTorch. Data loaders help users iterate through a dataset and create batches.

<details>
<summary><b>Hint</b>: More information about <a href="https://pytorch.org/tutorials/beginner/basics/data_tutorial.html"><code>DataLoader</code></a>.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
PyTorch provides a utility called <a href="https://pytorch.org/tutorials/beginner/basics/data_tutorial.html"><code>DataLoader</code></a> to help us create batches. It can work on a variety of underlying data structures, but in this assignment, we'll just apply it to the datasets you prepared previously.
<pre>
dl = DataLoader(your_dataset, batch_size=..., shuffle=...)
</pre>
The arguments here are as follows:
<ul>
<li><code>batch_size</code>: the number of instances in each batch.</li>
<li><code>shuffle</code>: whether or not we rearrange the instances randomly. It is common to shuffle instances while training.</li>
</ul>
When you have created a <code>DataLoader</code>, you can iterate through the dataset batch by batch:
<pre>for batch in dl:
   ... do something with each batch ...</pre>
</div>
</details>
&nbsp;

**Sanity check**: create a DataLoader, look at the first batch, and confirm that it corresponds to your expectations.
```
for batch in dl:
    print(batch)
    break
```

## Part 3: Defining the language model neural network

Define a neural network that implements an RNN-based language model. Use the skeleton provided in the class `A1RNNModel`. It should include the following layers:

- an *embedding layer* that maps token integers to floating-point vectors,
- an *recurrent layer* implementing some RNN variant (we suggest [`nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) or [`nn.GRU`](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html), and it is best to avoid the "basic" `nn.RNN`),
- an *output layer* (or *unembedding layer*) that computes (the logits of) a probability distribution over the vocabulary.

Once again, we base our implementation on the HuggingFace Transformers library, to exemplify how models are defined when we use this library. Specifically, note that
- The model hyperparameters are stored in a configuration object `A1RNNModelConfig` that inherits from HuggingFace's `PretrainedConfig`;
- The neural network class inherits from HuggingFace's `PreTrainedModel` rather than PyTorch's `nn.Module`.

When you set up your model, you should use the hyperparameters stored in the `A1RNNModelConfig`.

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
<p>
Take a look at the documentation of one of the RNN types in PyTorch. For instance, here is the documentation of <a href="https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"><code>nn.LSTM</code></a>. In particular, look at the section called <b>Outputs</b>. It is important to note here that all types of RNNs return <b>two</b> outputs when you call them in the forward pass. In this assignment, you will need the <b>first</b> of these outputs, which correspond to the RNN's output for each <em>token</em>. (The other outputs are the <em>layer-wise</em> outputs.)
</p>
<pre>
class MyRNNBasedLanguageModel(nn.Module):
  def __init__(self, ... ):
    super().__init__()
    ... initialize model components here ...
    
  def forward(self, batch):
    embedded = ... apply the embedding layer ...
    rnn_out, _ = self.rnn(embedded)
    ... do the rest ...
</pre></div>
</details>
&nbsp;

**Sanity check**: carry out the following steps:
- Create an integer tensor of shape 1x*N* where *N* is the length of the sequence. It doesn't matter what the integers are except that they should be less than the vocabulary size. (Alternatively, take one instance from your training set.)
- Apply the model to this input tensor. It shouldn't crash here.
- Make sure that the shape of the returned output tensor is 1x*N*x*V* where *V* is the size of the vocabulary. This output corresponds to the logits of the next-token probability distribution, but it is useless at this point because we haven't yet trained the model.


## Part 4: Training the model

We will now put all the pieces together and implement the code to train the language model.

Similarly to Part 1, we will mimic the functionality of the HuggingFace Transformers library. The [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) is the main utility the Transformers library provides to handle model training, and it provides a variety of complex functionality including multi-GPU training and many other bells and whistles. In our case, we will just implement a basic training loop.

Starting from the skeleton Python code, your task now is to complete the missing parts in the method `train` in the class `A1Trainer`. 

The missing parts you need to provide are
- Setting up the optimizer, which is the PyTorch utility that updates model parameters during the training loop. The optimizer typically implements some variant of [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). We recommend [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html), which is used to train most LLMs.
- Setting up the `DataLoader`s for the training and validation sets. The datasets are provided as inputs, and you can simply create the `DataLoader`s as in Part 2.
- The training loop itself, which is where most of your work will be done.

Hyperparameters that control the training should be stored in a [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) object. HuggingFace defines a large number of such hyperparameters but you only need to consider a few of them. The skeleton code includes a hint that lists the relevant hyperparameters.

The training loop should look more or less like a regular PyTorch training loop (see the hint in the code). There are a few non-trivial things to keep in mind when training an autoregressive language model (as opposed to training e.g. classifiers or regression models). We will discuss these points in the following three hints:
<details>
<summary><b>Hint</b>: the output tensor is the input tensor, shifted one step to the right.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
For instance, let's say our training text is <em>This is great !</em> (in practice, the words will be integer-coded).
That means that at the first word (<em>This</em>), we want the model to predict the second word (<em>is</em>). At the second word, the goal is to predict <em>great</em>, and so on.

So when you process a batch in the training loop, you should probably split it into an input and an output part:
<pre>
input_tokens = input_ids[:, :-1]
output_tokens = input_ids[:, 1:]
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
&nbsp;

While developing the code, we advise you to work with very small datasets until you know it doesn't crash, and then use the full training set. Monitor the cross-entropy loss (and/or the perplexity) over the training: if the loss does not decrease while you are training, there is probably an error. For instance, if the learning rate is set to a value that is too large, the loss values may be unstable or increase.

## Step 5: Evaluation and analysis

### Predicting the next word

Take some example context window and use the model to predict the next word.
- Apply the model to the integer-encoded context window. As usual, this gives you (the logits of) a probability distribution over your vocabulary.
- Use <a href="https://pytorch.org/docs/stable/generated/torch.argmax.html"><code>argmax</code></a> to find the index of the highest-scoring item, or <a href="https://pytorch.org/docs/stable/generated/torch.topk.html"><code>topk</code></a> to find the indices and scores of the *k* highest-scoring items.
- Apply the inverse vocabulary encoder (that you created in Step 2) so that you can understand what words the model thinks are the most likely in this context.

**Make sure that one or more examples of next-word prediction is printed by your Python program and included in the submitted output file.**

### Quantitative evaluation

The most common way to evaluate language models quantitatively is the [perplexity](https://huggingface.co/docs/transformers/perplexity) score on a test dataset. The better the model is at predicting the actually occurring words, the lower the perplexity. This quantity is formally defined as follows:

$$\text{perplexity} = 2^{-\frac{1}{m}\sum_{i=1}^m \log_2 P(w_i | c_i)}$$

In this formula, *m* is the number of words in the dataset, *P* is the probability assigned by our model, <em>w<sub>i</sub></em> and <em>c<sub>i</sub></em> the word and context window at each position.

Compute the perplexity of your model on the validation set. The exact value will depend on various implementation choices you have made, how much of the training data you have been able to use, etc. Roughly speaking, if you get perplexity scores around 700 or more, there are probably problems. Carefully implemented and well-trained models will probably have perplexity scores in the range of 200&ndash;300.

<details>
<summary><b>Hint</b>: An easy way to compute the perplexity in PyTorch.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
As you can see in the formula, the perplexity is an exponential function applied to the mean of the negative log probability of each token.
You are probably already computing the <em>cross-entropy loss</em> as part of your training loop, and this actually computes what you need here.

The perplexity is traditionally defined in terms of logarithms of base 2. However, we will get the same result regardless of what logarithmic base we use. So it is OK to use the natural logarithms and exponential functions, as long as we are consistent: this means that we can compute the perplexity by applying <code>exp</code> to the mean of the cross-entropy loss over your batches in the validation set.
</div>
</details>

If you have time for exploration, investigate the effect of model hyperparameters and training settings on the model's perplexity.

**Make sure that the perplexity computed on the validation set is printed by your Python program and included in the submitted output file.**

### Optional task: Inspecting the learned word embeddings

It is common to say that neural networks are "black boxes" and that we cannot fully understand their internal mechanics, especially as they grow larger and structurally more complex. The research area of model interpretability aims to develop methods to help us reason about the high-level functions the models implement.

We will briefly investigate the [embeddings](https://en.wikipedia.org/wiki/Word_embedding) that your model learned while you trained it.
If we have successfully trained a word embedding model, an embedding vector stores a crude representation of "word meaning", so we can reason about the learned meaning representations by investigating the geometry of the vector space of word embeddings.
The most common way to do this is to look at nearest neighbors in the vector space: intuitively, if we look at some example word, its neighbors should correspond to words that have a similar meaning.

Select some example words (e.g. `"sweden"`) and look at their nearest neighbors in the vector space of word embeddings. Does it seem that the nearest neighbors make sense?

<details>
<summary><b>Hint</b>: Example code for computing nearest neighbors.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
The following code shows how to compute the nearest neighbors in the embedding space of a given word. Depending on your implementation, you may need to change some details. Here, <code>emb</code> is the <code>nn.Embedding</code> module of your language model, while <code>voc</code> and <code>inv_voc</code> are the string-to-integer and integer-to-string mappings you created in Step 2.
<pre>
def nearest_neighbors(emb, voc, inv_voc, word, n_neighbors=5):

    # Look up the embedding for the test word.
    test_emb = emb.weight[voc[word]]
    
    # We'll use a cosine similarity function to find the most similar words.
    sim_func = nn.CosineSimilarity(dim=1)
    cosine_scores = sim_func(test_emb, emb.weight)
    
    # Find the positions of the highest cosine values.
    near_nbr = cosine_scores.topk(n_neighbors+1)
    topk_cos = near_nbr.values[1:]
    topk_indices = near_nbr.indices[1:]
    # NB: the first word in the top-k list is the query word itself!
    # That's why we skip the first position in the code above.
    
    # Finally, map word indices back to strings, and put the result in a list.
    return [ (inv_voc[ix.item()], cos.item()) for ix, cos in zip(topk_indices, topk_cos) ]
</pre>
</div>
</details>

Optionally, you may visualize some word embeddings in a two-dimensional plot (use a notebook while plotting or save the generated plot to a file via `plt.savefig`).
<details>
<summary><b>Hint</b>: Example code for PCA-based embedding scatterplot.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<pre>
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
def plot_embeddings_pca(emb, inv_voc, words):
    vectors = np.vstack([emb.weight[inv_voc[w]].cpu().detach().numpy() for w in words])
    vectors -= vectors.mean(axis=0)
    twodim = TruncatedSVD(n_components=2).fit_transform(vectors)
    plt.figure(figsize=(5,5))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.02, y, word)
    plt.axis('off')

plot_embeddings_pca(model[0], prepr, ['sweden', 'denmark', 'europe', 'africa', 'london', 'stockholm', 'large', 'small', 'great', 'black', '3', '7', '10', 'seven', 'three', 'ten', '1984', '2005', '2010'])
</pre>
</div>
</details>