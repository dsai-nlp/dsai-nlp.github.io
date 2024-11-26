---
layout: page
title: 'DAT450/DIT247: Programming Assignment 4: Comparing fine-tuning methods'
permalink: /courses/dat450/assignment4/
description:
nav: false
nav_order: 4
---

# DAT450/DIT247: Programming Assignment 4: Comparing fine-tuning methods

In this assignment, TODO

### Pedagogical purposes of this assignment
- x
- y

<details>
<summary><b>Hint</b>: Testing.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
Testing.
</div>
</details>

### Requirements

Please submit your solution in [Canvas](). **Submission deadline**: November XYZ.

Submit a notebook containing your solution to the programming tasks described below. This is a pure programming assignment and you do not have to write a technical report or explain details of your solution in the notebook: there will be a separate individual assignment where you will answer some conceptual questions about what you have been doing here.

### Acknowledgement

This assignment is a lightly modified version of a similar assignment by Marco Kuhlmann.

## Step 0: Preliminaries

### Libraries
In this assignment, we will rely on a set of libraries from the [HuggingFace](https://huggingface.co/) community:
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Datasets](https://huggingface.co/docs/datasets/index)
- [Evaluate](https://huggingface.co/docs/evaluate/en/index)

Make sure all libraries are installed in your environment. 
If you use Colab, you will need to install Datasets and Evaluate, while Transformers is included in the pre-installed environment.

### Getting the files

The data for this lab comes from the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). The full dataset consists of 50,000 highly polar movie reviews collected from the Internet Movie Database (IMDB). Here, we use a random sample consisting of 2,000 reviews for training and 500 reviews for evaluation.

## Step 1: Full fine-tuning

### Preprocessing

<details>
<summary><b>Hint</b>: Creating a Dataset.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<pre>
from datasets import load_dataset
imdb_dataset = load_dataset('csv', data_files = {'train': 'path/to/train.csv', 'eval': 'path/to/eval.csv'})
</pre>
</div>
</details>

<details>
<summary><b>Hint</b>: Applying a tokenizer to a Dataset.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<pre>
def tokenize_helper(batch):
    return tokenizer(batch['review'], padding=True, truncation=True)
tokenized_imdb_dataset = imdb_dataset.map(tokenize_helper, batched=True)
</pre>
</div>
</details>

### Defining your model

<pre>
pretrained_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
</pre>

### Counting the number of trainable parameters

Define a function `count_trainable_parameters` that computes the number of floating-point numbers that a given model will update during training.

- The methods `.parameters()` and `.named_parameters()` return a sequence of tensors containing the model parameters.
-
- 

<details>
<summary><b>Hint</b>: Applying a tokenizer to a Dataset.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
</div>
</details>

**Sanity check**: The number of trainable parameters for the model above should be 66955010.

### Creating a Trainer

### Running the Trainer

## Step 2: Tuning the final layers only

## Interlude: Replacing layers

## Step 3: Fine-tuning with LoRA





