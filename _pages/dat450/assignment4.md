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

The data we use in this assignment is a subset of the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). The full dataset consists of 50,000 highly polar movie reviews collected from the Internet Movie Database (IMDB). We use a random sample consisting of 2,000 reviews for training and 500 reviews for evaluation.

## Step 1: Full fine-tuning

In this assignment, we will use a compressed version of BERT called [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert).
We'll use the uncased version: that is, the tokenizer will not distinguish uppercase and lowercase.

In the HuggingFace utilities that require you to specify a model name, you should use `distilbert-base-uncased`.

### Preprocessing

[Create a Dataset](https://huggingface.co/docs/datasets/create_dataset) by loading the training and evaluation CSV files you previously downloaded.
<details>
<summary><b>Hint</b>: Creating a Dataset.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<pre>
from datasets import load_dataset
imdb_dataset = load_dataset('csv', data_files = {'train': 'path/to/train.csv', 'eval': 'path/to/eval.csv'})
</pre>
</div>
</details>

Load the pre-trained tokenizer using `AutoTokenizer` and apply it to the Dataset.

<details>
<summary><b>Hint</b>: Applying a tokenizer to a Dataset.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
It's easiest if we create a helper function that applies the tokenizer to the right Dataset columns and with the right parameters.
<pre>
def tokenize_helper(batch):
    return tokenizer(batch['review'], padding=True, truncation=True)
tokenized_imdb_dataset = imdb_dataset.map(tokenize_helper, batched=True)
</pre>
This step will create new Dataset columns `input_ids` and `attention_mask`.
</div>
</details>

**Note**: you may receive some warnings caused by parallelism in the tokenizer. To get rid of the warnings, you can use the following workaround.

<pre>
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
</pre>

### Creating your classification model for fine-tuning

Use the HuggingFace utility `AutoModelForSequenceClassification` to set up a model that you can fine-tune. Use the `from_pretrained` method with the model name set as above, and `num_labels=2` (because we have two-class classification task). This method carries out the following steps:
- It loads the pre-trained DistillBERT model from the HuggingFace repository (or from a cached file, if you have used the model before).
- It sets up untrained layers to map from the DistillBERT output to the two class labels. They will be trained during the fine-tuning process below.

**Sanity check**: Print the model in a notebook cell. You should see a visual representation of layers the model consists of. You should see the DistillBERT model including embedding layers and Transfomer layers. At the bottom of the list of layers, you should see two layers called `pre_classifier` and `classifier`, which are the newly created classification layers.

### Counting the number of trainable parameters

Define a function `count_trainable_parameters` that computes the number of floating-point numbers that a given model will update during training.

- The methods `.parameters()` and `.named_parameters()` return a sequence of tensors containing the model parameters.
- When counting the **trainable** parameters, you should only include those tensors where `requires_grad` is `True`. That is: we want to exclude tensors containing parameters we will not update during training.

**Sanity check**: The number of trainable parameters for the model above should be 66955010.

### Preparing for training

The class `TrainingArguments` defines some parameters controlling the training process. We'll mostly use default values here. You only need to set the following parameters:
- `output_dir`: the name of some directory where the `Trainer` will keep its file.
- `num_train_epochs`: the number of training epochs.
- `eval_strategy`: set this to `epoch` to see evaluation scores after each epoch.

In addition, we need to define a helper function that will be used for evaluation after each epoch. We use a utility from the Evaluate library for this:

<pre>
import evaluate

accuracy_scorer = evaluate.load('accuracy')

def evaluation_helper(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_scorer.compute(predictions=predictions, references=labels)
</pre>

### Training the model

Import `Trainer` from the `transformers` library. Create a `Trainer` using the following arguments:
- `model`: the model that you are fine-tuning;
- `args`: the training arguments you defined above;
- `train_dataset`: the `train` section of your tokenized `Dataset`;
- `eval_dataset`: the `eval` section of your tokenized `Dataset`;
- `compute_metrics`: the evaluation helper function you defined above.

Run the fine-tuning process by calling `train()` on your `Trainer`.
This will train for the specified number of epochs, computing loss and accuracy after each epoch.

After training, you may call `save_model` on the `Trainer` to save the model's parameters. In this way, you can reload it later without having to retrain it.

<details>
<summary><b>Hint</b>: Avoiding accidental model reuse.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
It is probably a good idea to re-create the model (using <code>AutoModelForSequenceClassification.from_pretrained</code>) before each time you train it. Otherwise, you may accidentally train a model that has already been trained.
</div>
</details>

## Step 2: Tuning the final layers only

Even with a minimal model such as DistilBERT, fine-tuning the full model is rather time-consuming. We will now consider fine-tuning approaches where we only work with a subset of the model's parameters.

Set up the model once again. Disable gradient computation for all parameter tensors except those that are trained from scratch. That is: the two layers in the classification head will be updated during training, while the DistilBERT model will be kept fixed.

**Sanity check**: The number of trainable parameters for this model should be 592130.

<details>
<summary><b>Hint</b>: Avoiding accidental model reuse, again!</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
Once again, we recommend that you re-create the model using <code>AutoModelForSequenceClassification.from_pretrained</code> before this step, so that you don't accidentelly work with the model that you fine-tuned in Step 1.
</div>
</details>

<details>
<summary><b>Hint</b>: How to disable gradient computation for a parameter tensor.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<p>
For a parameter tensor in a model, we can set the attribute <code>requires_grad</code> to <code>False</code>, which means that during backpropagation, gradients will not be computed with respect to these parameters. So the training process will not change these parameters.
</p>

<p>
To find the parameter tensors to switch off, you can either 1) go into the <code>distilbert</code> component and iterate through its <code>parameters</code>, or 2) go through all the model's named parameters, and switch off all parameter tensors except <code>classifier</code> and <code>pre_classifier</code>.
</p>
</div>
</details>

Train this model and compare the training speed and classification accuracy to the results from Step 1.

## Step 3: Fine-tuning with LoRA

### Utilities for modifying models

Define a function `extract_qv_layers` that extracts the query and value linear layers from all Transformer blocks in a DistilBERT model.
Return a dictionary that maps the component name to the corresponding linear layer.

<details>
<summary><b>Hint</b>: How to access the query and value linear layers.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<p>
As we saw earlier, the DistilBERT model consists of a hierarchy of nested submodules. Each of these can be addressed by a fully-qualified string name.
</p>

<p>
You can use get_submodule() to retrieve a layer by a string name. For instance, <code>'distilbert.transformer.layer.0.attention.q_lin'</code> refers to the Q part of Transformer layer 0.
</p>

<p>
It's OK to hard-code this part, so that you just enumerate the Q and V parts of all layers here.
</p>
</div>
</details>

**Sanity check**: If you apply this on a DistillBERT model, the result should contain 12 named linear layers.

Then, 

<pre>
def replace_layers(model, named_layers):
    for name, layer in named_layers.items():
        components = name.split('.')
        submodule = model
        for component in components[:-1]:
            submodule = getattr(submodule, component)
        setattr(submodule, components[-1], layer)
</pre>

### Implementing the LoRA layer

<pre>
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, W, rank=12, alpha=24):
        super().__init__()
        # TODO: Add your code here

    def forward(self, x):
        # TODO: Replace the next line with your own code
        raise NotImplementedError
</pre>

### Fine-tuning with LoRA

Set up a model where you replace the query and value linear layers with LoRA layers. Use the following steps:
- First use `extract_qv_layers` to get the relevant linear layers.
- Each of the linear layers in the returned dictionary should be wrapped inside a LoRA layer.
- Then use `replace_layers` to put them back into the model.

**Sanity check**: Use your function <code>count_trainable_parameters</code>. The number of trainable parameters should be less than in Step 1 but more than in Step 2. The exact number will depend on the rank.

Train this model and compare the training speed and classification accuracy to the results from Steps 1 and 2.