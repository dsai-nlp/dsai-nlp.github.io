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
- We will now use a *Transformer* instead of the recurrent neural network we had previously.
- In this assignment, we will also use our language model to generate texts.

### Pedagogical purposes of this assignment
- Understanding the Transformer architecture in details, when used for language modeling.
- Understanding text-generating algorithms.

### Requirements

Please submit your solution in [Canvas](https://canvas.chalmers.se/courses/36909/assignments/117615). **Submission deadline**: November 17.

Submit Python files containing your solution to the programming tasks described below. In addition, to save time for the people who grade your submission, please submit a text file containing the outputs printed out by your Python program; read the instructions carefully so that the right outputs are included. (Most importantly: the perplexity evaluated on the validation set, and the generated texts you have created in the last section.)

This is a pure programming assignment and you do not have to write a technical report or explain details of your solution: there will be a separate individual assignment where you will answer some conceptual questions about what you have been doing here.
## Step 0: Preliminaries

Make sure you have access to your solution for Programming Assignment 1 since you will reuse the tokenization and the training loop. (Optionally, use HuggingFace's `Trainer` instead.)

On Minerva, copy the skeleton from `/data/courses/2025_dat450_dit247/assignments/a2/A2_skeleton.py`.
This skeleton contains stub classes for all Transformer components, as well as a complete implementation of the RoPE positional representation (copied and somewhat simplified from the HuggingFace library).

## Step 1: Setting up a Transformer neural network

The main effort in this assignment is the reimplementation of a Transformer architecture. Specifically, we will mimic the architecture of the [OLMo 2](https://docs.allenai.org/release_notes/olmo-release-notes) language model, released by the [Allen AI institute](https://allenai.org/about) at the University of Washington.

The figure below shows the design of the OLMo 2 Transformer. We will reimplement the MLP component and the multi-head attention (and optionally the normalizer as well), and then assemble all the pieces into a full Transformer stack.

<img src="https://raw.githubusercontent.com/ricj/dsai-nlp.github.io/refs/heads/master/_pages/dat450/olmo2_overview.svg" alt="Olmo2 overview" style="width:10%; height:auto;">

**Implementation note:** To be 100% compatible with the OLMo 2 implementation, note that all the `nn.Linear` inside of all layers are without bias terms (`bias=False`). This includes query, key, value, and output projections inside attention layers, all parts of the MLP layers, and the unembedding layer. If you solve the optional task at the end where you copy the weights of a pre-trained model into your implementation, then it is important that all layers are identical in structure.

### Configuration

Similarly to Assignment 1, the model hyperparameters you need for this assignment will be stored in a configuration object `A2ModelConfig`, which inherits from HuggingFace's `PretrainedConfig`. This configuration will be passed into `__init__` of all the Transformer's components.

### MLP layer

OLMo 2 uses an MLP architecture called SwiGLU, which was introduced in [this paper](https://arxiv.org/pdf/2002.05202). (In the paper, this type of network is referred to as FFN<sub>SwiGLU</sub>, described on page 2, Equation 6. Swish<sub>1</sub> corresponds to PyTorch's [SiLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html) activation.) The figure below shows the architecture visually; in the diagram, the ⊗ symbol refers to element-wise multiplication.

<img src="https://raw.githubusercontent.com/ricj/dsai-nlp.github.io/refs/heads/master/_pages/dat450/swiglu.svg" alt="SwiGLU" style="width:10%; height:auto;">

The relevant hyperparameters you need to take into account here are `hidden_size` (the dimension of the input and output) and `intermediate_size` (the dimension of the intermediate representations).

**Sanity check.**

Create an untrained MLP layer. Create some 3-dimensional tensor where the last dimension has the same size as `hidden_size` in your MLP. If you apply the MLP to the test tensor, the output should have the same shape as the input.

### Normalization

To stabilize gradients during training, deep learning models with many layers often include some *normalization* (such as batch normalization or layer normalization). Transformers typically includes normalization layers at several places in the stack.
OLMo 2 uses a type of normalization called [Root Mean Square layer normalization](https://arxiv.org/pdf/1910.07467).

You can either implement your own normalization layer, or use the built-in [`RMSNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) from PyTorch. In the PyTorch implementation, `eps` corresponds to `rms_norm_eps` from our model configuration, while `normalized_shape` should be equal to the hidden layer size. The hyperparameter `elementwise_affine` should be set to `True`, meaning that we include some learnable weights in this layer instead of a pure normalization.

If you want to make your own layer, the PyTorch documentation shows the formula you should implement. (The $$\gamma_i$$ parameters are the learnable weights.)

**Sanity check.**

You can test this in the same way as you tested the MLP previously.

### Multi-head attention

Now, let's turn to the tricky part!

The smaller versions of the OLMo 2 model, which we will follow here, use the same implementation of *multi-head attention* as the original Transformer, plus a couple of additional normalizers. (The bigger OLMo 2 models use [grouped-query attention](https://sebastianraschka.com/llms-from-scratch/ch04/04_gqa/) rather than standard MHA; GQA is also used in various Llama, Qwen and some other popular LLMs.)

The figure below shows a high-level overview of the pieces we will have to put together. (In the figure, the four *W* blocks are `nn.Linear`, and RN means RMSNorm.)

<img src="https://raw.githubusercontent.com/ricj/dsai-nlp.github.io/refs/heads/master/_pages/dat450/mha.svg" alt="MHA" style="width:10%; height:auto;">

**Hyperparameters:** The hyperparameters you will need to consider when implementing the MHA are 
`hidden_size` which defines the input dimensionality as in the MLP and normalizer above, and
`num_attention_heads` which defines the number of attention heads. **Note** that `hidden_size` has to be evenly divisible by `num_attention_heads`. (Below, we will refer to `hidden_size // num_attention_heads` as the head dimensionality $$d_h$$.)

**Defining MHA components.** In `__init__`, define the `nn.Linear` components (square matrices) that compute query, key, and value representations, and the final outputs. (They correspond to what we called $$W_Q$$, $$W_K$$, $$W_V$$, and $$W_O$$ in [the lecture on Transformers](https://www.cse.chalmers.se/~richajo/dat450/lectures/l4/m4_2.pdf).) OLMo 2 also applies layer normalizers after the query and key representations.

**MHA computation, step 1.** The `forward` method takes two inputs `hidden_states` and `rope_rotations`. The latter contains the precomputed rotations required for RoPE. (The section **The complete Transformer stack** below explains where they come from.)

Continuing to work in  `forward`, now compute query, key, and value representations; don't forget the normalizers after the query and key representations.

Now, we need to reshape the query, key, and value tensors so that the individual attention heads are stored separately. Assume your tensors have the shape $$ (b, m, d) $$, where $$ b $$ is the batch size, $$ m $$ the text length, and $$ d $$ the hidden layer size. We now need to reshape and transpose so that we get $$ (b, n_h, m, d_h) $$ where $$ n_h $$ is the number of attention heads and $$ d_h $$ the attention head dimensionality. Your code could be something like the following (apply this to queries, keys, and values):

```
q = q.view(b, m, n_h, d_h).transpose(1, 2)
```

Now apply the RoPE rotations to the query and key representations. Use the utility function `apply_rotary_pos_emb` provided in the code skeleton and just provide the `rope_rotations` that you received as an input to `forward`. The utility function returns the modified query and key representations.

**Sanity check step 1.**
Create an untrained MHA layer. Create some 3-dimensional tensor where the last dimension has the same size as `hidden_size`, as you did in the previous sanity checks. Apply the MHA layer with what you have implemented so far and make sure it does not crash. (It is common to see errors related to tensor shapes here.)

**MHA computation, step 2.** Now, implement the attention mechanism itself. 
We will explain the exact computations in the hint below, but conveniently enough PyTorch's [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (with `is_causal=True`) implements everything that we have to do here.  Optionally, implement your own solution.

<details>
<summary><b>Hint</b>: Some advice if you want to implement your own attention.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
In that case, the <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">documentation of the PyTorch implementation</a> includes a piece of code that can give you some inspiration and that you can simplify somewhat.

Assuming your query, key, and value tensors are called \(q\), \(k\), and \(v\), then the computations you should carry out are the following. First, we compute the <em>attention pre-activations</em>, which are compute by multiplying query and key representations, and scaling:

$$
\alpha(q, k) = \frac{q \cdot k^{\top}}{\sqrt{d_h}}
$$

The transposition of the key tensor can be carried out by calling <code>k.transpose(-2, -1)</code>.

Second, add a *causal mask* to the pre-activations. This mask is necessary for autoregressive (left-to-right) language models: this is so that the attention heads can only consider tokens before the current one. The mask should have the shape \((m, m)\); its lower triangle including the diagonal should be 0 and the upper triangle \(-\infty\). Pytorch's <a href="https://docs.pytorch.org/docs/stable/generated/torch.tril.html"><code>tril</code></a> or <a href="https://docs.pytorch.org/docs/stable/generated/torch.triu.html"><code>triu</code></a> can be convenient here.

Then apply the softmax to get the attention weights.

$$
A(q, k) = \text{softmax}(\alpha(q, k) + \text{mask})
$$

Finally, multiply the attention weights by the value tensor and return the result.

$$
\text{Attention}(q, k, v) = A(q, k) \cdot v
$$
</div>
</details>

**MHA computation, step 3.** Now, we need to combine the results from the individual attention heads. We first flip the second and third dimensions of the tensor (so that the first two dimensions correspond to the batch length and text length), and then reshape into the right shape.
```
attn_out = attn_out.transpose(1, 2).reshape(b, m, d)
```
Then compute the final output representation (by applying the linear layer we called $$W_O$$ above) and return the result.

**Sanity check steps 2 and 3.**
Once again create a MHA layer for testing and apply it to an input tensor of the same shape as before. Assuming you don't get any crashes here, the output should be of the same shape as the input. If it crashes or your output has the wrong shape, insert `print` statements along the way, or use an editor with step-by-step debugging, to check the shapes at each step.

### The full Transformer decoder layer

After coding up the multi-head attention, everything else is just a simple assembly of pieces!
The figure below shows the required components in a single Transformer decoder layer.

<img src="https://raw.githubusercontent.com/ricj/dsai-nlp.github.io/refs/heads/master/_pages/dat450/fullblock.svg" alt="fullblock" style="width:10%; height:auto;">

In the constructor `__init__`, create the components in this block, taking the model configuration into account.
As shown in the figure, a Transformer layer should include an attention layer and an MLP, with normalizers. In `forward`, connect the components to each other; remember to put residual connections at the right places.

<details>
<summary><b>Hint</b>: Residual connections in PyTorch.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
Assuming your input is called <code>h_old</code>, then a residual connection is implemented via a straightforward addition.
<pre>
h_new = do_something(h_old) 
out = h_new + h_old
</pre>
</div>
</details>

**Sanity check.** Carry out the usual sanity check to see that the shapes are correct and there are no crashes.

### The complete Transformer stack

Now, set up the complete Transformer stack including embedding, top-level normalizer, and unembedding layers. (You may look at the figure presented previously.)
The embedding and unembedding layers will be identical to what you had in Programming Assignment 1 (except that the unembedding layer should not use bias terms, as mentioned in the beginning).

<details>
<summary><b>Hint</b>: Use a <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html"><code>ModuleList</code></a>.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
Put all the Transformer blocks in a <code>ModuleList</code> instead of a plain Python list. The <code>ModuleList</code> makes sure your parameters are registered so that they are included when you compute the gradients.
</div>
</details>

<details>
<summary><b>Hint</b>: Creating and applying the RoPE embeddings.</summary>
<div style="margin-left: 10px; border-radius: 4px; background: #ddfff0; border: 1px solid black; padding: 5px;">
Create the <code>A2RotaryEmbedding</code> in <code>__init__</code>, as already indicated in the code skeleton. Then in <code>forward</code>, first create the rotations (again, already included in the skeleton). Then pass the rotations when you apply each Transformer layer.
</div>
</details>

**Sanity check.** Now, the language model should be complete and you can test this in the same way as in Programming Assignment 1. Create a 2-dimensional *integer* tensor and apply your Transformer to it. The result should be a 3-dimensional tensor where the last dimension is equal to the vocabulary size.

## Step 2: Training the language model

In Assignment 1, you implemented utilities to tokenize the text, load the documents, and to handle training and validation. Your Transformer language model should be possible to use as a drop-in replacement for the RNN-based model you had in that assignment.

**Alternative solution.** Use a HuggingFace Trainer.

Select some suitable hyperparameters (number of Transformer layers, hidden layer size, number of attention heads).
For this assignment, you are recommended to use a small Transformer (e.g. a couple of layers).
Then run the training function and compute the perplexity on the validation set as in the previous assignment. 

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

Here are a few example prompts that could be interesting to try:
<pre>
'In natural language processing, a Transformer'
'Is Stockholm the capital of Sweden? Answer yes or no. The answer is'
'Write a Python program that reverses a list.'
</pre>

### Comparing to a pre-trained Transformer

Your language model will probable be able to generate texts that look somewhat like English, but rather bland and nonsensical. As an alternative, let's load the pre-trained OLMo-2 model (the 1 billion-parameter version). We have downloaded a copy to Minerva to save you some download time. Here, 

```
from transformers import AutoTokenizer, AutoModelForCausalLM
local_dir = '/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B'
tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)
```

**Note:** when you apply this model, the return value is a [`CausalLMOutputWithPast`](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) object, not just the logits. This object has a field called `logits`. Otherwise, you should be able to use the pre-trained model in your generation algorithm.

Try the test examples once again with the pre-trained model and note the differences. In the reflection questions, there will be some questions about these differences.

Note that this is a pure language model (like the one you trained) and it has not been *instruction-tuned*. That is: it has not been post-trained to allow interactive chatting.

**Optional task.** To verify that your implementation is identical to the Olmo 2 model, copy the weight tensors from the pre-trained model into an instance of your own implementation, and verify that you get exactly the same results.