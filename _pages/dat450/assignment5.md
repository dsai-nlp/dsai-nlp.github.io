---
layout: page
title: 'DAT450/DIT247: Programming Assignment 5: Retrieval-augmented text generation'
permalink: /courses/dat450/assignment5/
description:
nav: false
nav_order: 4
---

# DAT450/DIT247: Programming Assignment 5: Retrieval-augmented text generation
In this assignment we will build our own RAG pipeline using LangChain.

### Pedagogical purposes of this assignment
- Get an understanding of how RAG can be used within NLP.
- Learn how to use LangChain to build NLP applications.
- Get an understanding for the challenges and use cases of RAG.

### Requirements
Please submit your solution in [Canvas](https://chalmers.instructure.com/courses/31739/assignments/98457). **Submission deadline:** December 13.

Submit a notebook containing your solution to the programming tasks described below. This is a pure programming assignment and you do not have to write a technical report: there will be a separate individual assignment where you will answer some conceptual questions about what you have been doing here. However, you are welcome to write down your thoughts in this notebook, while you will not be assessed on them here.

## Step 0: Get the datasets
You will be working with the [PubMedQA dataset](https://github.com/pubmedqa/pubmedqa) described in this [paper](https://aclanthology.org/D19-1259.pdf). The dataset has been created based on medical research papers from [PubMed](https://pubmed.ncbi.nlm.nih.gov/), you can read more about it in the linked paper.

Use the following code to get the dataset for the assignment.

```bash
wget https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json
```

### Collect two datasets from the downloaded data

We collect two datasets:
- 'questions': the questions with corresponding gold long answer, gold document ID, and year.
- 'documents': the abstracts (contexts+long_answer concatenated), and year.

```python
import pandas as pd
tmp_data = pd.read_json("/content/ori_pqal.json").T
# some labels have been defined as "maybe", only keep the yes/no answers
tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

documents = pd.DataFrame({"abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS+[row.LONG_ANSWER]), axis=1),
             "year": tmp_data.YEAR})
questions = pd.DataFrame({"question": tmp_data.QUESTION,
             "year": tmp_data.YEAR,
             "gold_label": tmp_data.final_decision,
             "gold_context": tmp_data.LONG_ANSWER,
             "gold_document_id": documents.index})
```

For an example of a query:

```python
questions.iloc[0].question
```

For an example of a document to leverage for the queries:

```python
documents.iloc[0].abstract
```

> Note that we will increase the difficulty of the pipeline in the sense that it needs to find the relevant document on its own. E.g. for question 0 we will not directly give the model abstract 0.

## Step 1: Configure your LangChain LM

Define a language model that will act as the generative model in your RAG pipeline. You can for example use the [HuggingFacePipeline](https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/) in LangChain to run models on your GPU. You can browse for different Hugging Face models on their [webpage](https://huggingface.co/models). A general guide on how to set up RAG pipelines in LangChain can be found [here](https://python.langchain.com/v0.1/docs/use_cases/chatbots/retrieval/#creating-a-retriever).

> You should be able to fit a model of at least a size of 1B parameters on the T4 GPUs available in Colab.

> Some interesting models (e.g. Llama 3.2) may require that you apply for access. This process is usually quite fast, while it may require that you create an account on Hugging Face (it is free). To use a gated model you need to generate a personal HF token and put it as a secret in your notebook (if using Colab). Make sure that the token has enabled "Read access to contents of all public gated repos you can access".

**Sanity check:** Prompt your LangChain model and confirm that it returns a reasonable output.

## Step 2: Set up the document database and retriever

### Step 2.1: Embedding model
First, you need a model to embed the documents in the retrieval corpus. Here, we recommend using the [HuggingFaceEmbeddings](https://api.python.langchain.com/en/latest/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html) function.

**Sanity check:** Pass a text passage to the embedding model and evaluate its shape. It should be of the shape (1, embedding_dim).


### Step 2.2: Chunking
Second, you need to chunk the documents in your retrieval corpus, as some likely are too long for the embedding model. Here, you can use the [RecursiveCharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/) as a start. The retrieval corpus is given by `documents.abstract`.

**Sanity check:** Print some samples from the result and check that it makes sense. This way, you might be able to get a feeling for a good chunk size.

### Step 2.3: Define a vector store and retriever
Third, you need a vector store to store the documents and corresponding embeddings (indeces). There are many document databases and retrievers to play around with. As a start, you can use the [Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/) vector store with cosine similarity as the distance metric. You can then define the retriever using something like the following:

```python
retriever = vector_store.as_retriever(...)
```

As a start, you might want the retriever to fetch only one document per prompt.

**Sanity check:** Query your vector store as follows and check that the results make sense:
```python
results = vector_store.similarity_search_with_score(
    "What is programmed cell death?", k=3
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
```

## Step 3: Define the full RAG pipeline

We are now ready to combine all previously defined components into a complete RAG pipeline. Define a prompt and set up your chain with the retriever and generator LM. Here, you might want to define a chain that also returns what document was retrieved, the [RunnableParallel](https://python.langchain.com/v0.1/docs/expression_language/primitives/parallel/) function can be used for this.

**Sanity check:** Take a question from your dataset and check whether the model seems to retrieve a relevant document, and answer in a reasonable fashion.

## Step 4: Evaluate the RAG pipeline on the dataset

- Evaluate your full RAG pipeline on the medical questions (`questions.question`) and corresponding gold labels (`questions.gold_label`). Since the gold labels can be casted to a binary variable (yes/no) you may use the f1 metric.
- Also evaluate your retriever by checking whether it managed to fetch passages from the gold document with ID given by `questions.gold_document_id`.
- As a baseline, run the same LM without context and compare the performance of the two setups. Did the retrieval help?
- Also, inspect some retrieved documents and corresponding model answers. Does the pipeline seem to work as intended?

## Step 5: Make improvements

After having observed the performance of your pipeline, you might have some ideas on how to improve it. Thanks to the abstraction level in LangChain, it should be quite easy to experiment with different improvements. Experiment with at least two types of improvements to your RAG pipeline that you find interesting. Make sure to document your experiments and the corresponding results.

Aspects that can be tinkered with are for example:
- the document chunker: some alternatives can be found [here](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/),
- prompt: a guide on prompt tuning can be found [here](https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates/),
- retriever: some alternatives can be found [here](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/),
- embedding model: some alternatives can be found [here](https://python.langchain.com/docs/integrations/text_embedding/),
- etc...