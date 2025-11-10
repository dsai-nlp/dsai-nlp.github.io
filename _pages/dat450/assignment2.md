---
layout: page
title: 'DAT450/DIT247: Programming Assignment 2: Transformer language models'
permalink: /courses/dat450/assignment2/
description:
nav: false
nav_order: 4
---

# DAT450/DIT247: Programming Assignment 2: Transformer language models

In this assignment, we extend the models we investigated in the previous assignment in two different ways:
- In the previous assignment, we used a model that takes a fixed number of previous words into account. Now, we will use a model capable of considering a variable number of previous words: a *recurrent neural network*. (Optionally, you can also investigate *Transformers*.)
- In this assignment, we will also use our language model to generate texts.

### Pedagogical purposes of this assignment
- Understanding the Transformer architecture in details, when used for language modeling.
- Understanding text-generating algorithms.

### Requirements

Please submit your solution in [Canvas](https://chalmers.instructure.com/courses/XX/assignments/YY). **Submission deadline**: November XX.

Submit a XX

## Step 0: Preliminaries

Make sure you have access to your solution for Programming Assignment 1 since you will reuse some parts.

Copy the skeleton from SOMEWHERE.

## Step 1: Setting up a Transformer neural network

The main effort in this assignment is the reimplementation of a Transformer architecture. Specifically, we will mimic the architecture of the [OLMo 2](https://docs.allenai.org/release_notes/olmo-release-notes) language model, released by the [Allen AI institute](https://allenai.org/about) at the University of Washington.

The figure below shows the design of the OLMo 2 Transformer. We will reimplement the MLP component and the multi-head attention (and optionally the normalizer as well), and then assemble all the pieces into a full Transformer stack.

<img src="https://raw.githubusercontent.com/ricj/dsai-nlp.github.io/refs/heads/master/_pages/dat450/olmo2_overview.svg" alt="Olmo2 overview" style="width:10%; height:auto;">

**Implementation note:** To be fully compatible with the OLMo 2 implementation, note that all the `nn.Linear` inside of all layers are bias-free (`bias=False`). This includes Q, K, V, and O projections inside attention layers, all parts of the MLP layers, and the unembedding layer. If you solve the optional task at the end where you copy the weights of a pre-trained model into your implementation, then it is important that all layers are identical in structure.

### Configuration

### MLP layer

Olmo 2 uses an MLP architecture called SwiGLU, which was introduced in [this paper](https://arxiv.org/pdf/2002.05202). (In the paper, this type of network is referred to as FFN<sub>SwiGLU</sub>, described on page 2, Equation 6. Swish<sub>1</sub> corresponds to PyTorch's [SiLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html) activation.) The figure below shows the architecture visually.

<img src="https://raw.githubusercontent.com/ricj/dsai-nlp.github.io/refs/heads/master/_pages/dat450/swiglu.svg" alt="SwiGLU" style="width:10%; height:auto;">

**Sanity check.**

### Normalization

To stabilize gradients during training, deep learning models with many layers often include some *normalization* (such as batch normalization or layer normalization). Transformers typically includes normalization layers at several places in the stack.

Olmo 2 uses a type of normalization called [Root Mean Square layer normalization](https://arxiv.org/pdf/1910.07467).

Here, you can either implement your own normalization layer, or use the built-in [`RMSNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) from PyTorch. In the PyTorch implementation, `eps` corresponds to `rms_norm_eps` from our model configuration, while `normalized_shape` should be equal to the hidden layer size. The hyperparameter `elementwise_affine` should be set to `True`, meaning that we include some learnable weights in this layer instead of a pure normalization.

If you want to make your own layer, the PyTorch documentation shows the formula you will have to implement. (The $\gamma_i$ parameters are the learnable weights.)

**Sanity check.**

### Multi-head attention

Let's take the trickiest part first!

It is OK to use PyTorch's [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) to compute the final step. (In that case, set `is_causal=True`.)

If you want to use your own implementation, the [documentation of the PyTorch implementation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) includes a piece of code that you can start from.

**Sanity check.**

### The full Transformer block

**Sanity check.**

### The complete Transformer stack

The embedding and unembedding layers will be identical to what you had in Programming Assignment 1 (except that the unembedding layer should be bias-free, as mentioned above).

## Step 2: Training the language model

**Alternative solution.** Use a HuggingFace Trainer.

Run the training function and compute the perplexity on the validation set as in the previous assignment.

## Step 3: Generating text

### Predicting the next word

As a starting point, we'll repeat the exercise from the first assignment where we see what the model predicts as the next word of a given sequence. For instance, for the sequence `he lives in san`, a well-trained model will typically predict the word `francisco`. The steps will typically be something like the following:

- Apply the model to the integer-encoded input text.
- Take the model's output at the last position (but make sure that you avoid an end-of-sentence dummy here).
- Use <a href="https://pytorch.org/docs/stable/generated/torch.argmax.html"><code>argmax</code></a> to find the index of the highest-scoring item.
- Apply the inverse vocabulary encoder so that you can understand what words the model thinks are the most likely in this context.

### Generating texts

Implement a random sampling algorithm as described in the recording ([video](https://youtu.be/QtwpM-OGOew), [pdf](http://www.cse.chalmers.se/~richajo/dat450/lectures/l3/l3_generating.pdf)). The function should take the following inputs:

- `model`: the language model that we use to predict the next token.
- `prompt`: the prompt that initializes the text generation.
- `max_length`: the maximal number of steps before terminating.
- `temperature`: controls the degree of randomness by scaling the predicted logits.
- `topk`: to implement top-K sampling, i.e. the next-token distribution is truncated so that it only includes the `topk` most probable tokens.

The text generation should proceed until it an end-of-text symbol has been generated, or for at most `max_length` steps.

<details>
<summary><b>Hint</b>: How to sample from the next-token distribution.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
<p>
The easiest option is probably to use <a href="https://pytorch.org/docs/stable/distributions.html#categorical"><code>torch.distributions.Categorical</code></a>.
<code>Categorical</code> is a probability distribution over a set of choices, each of which has its own probability. So this is equivalent to the case where we have a set of possible next tokens, with different probabilities.
</p>

<p>
The following code shows an example of how <code>Categorical</code> can be used. In your code, you will replace <code>example_logits</code> with the next-token distribution predicted by your language model.
</p>

<pre>
# Logits of the probabilities of 5 different choices.
example_logits = torch.tensor([0.0, 0.5, -0.2, 0.1, 0.05])
example_distr = Categorical(logits=example_logits)
sampled = example_distr.sample()
</pre>
</div>
</details>

<details>
<summary><b>Hint</b>: The <a href="https://pytorch.org/docs/stable/generated/torch.topk.html"><code>topk</code></a> function will be useful when you implement top-K sampling.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
This function takes a tensor as input and returns the <em>k</em> highest scores and their corresponding indices.
</div>
</details>

Run your generation algorithm with some different prompts and input parameters, and try to investigate the effects. In the reflection questions, you will be asked to summarize your impression of how texts are generated with different prompts and input parameters.

**Sanity check**: There are two ways to make this random sampling algorithm behave like *greedy decoding* (that is: there is no randomness, and the most likely next word is selected in each step). Run the function in these two ways and make sure you get the same output in both cases.

### Comparing to a pre-trained Transformer

```
from transformers import AutoTokenizer, AutoModelForCausalLM
local_dir = '/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B'
tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)
```

Note that this

**Optional task.** To verify that your implementation is identical to the Olmo 2 model, copy the weight tensors from the pre-trained model into an instance of your own implementation, and verify that you get exactly the same results.