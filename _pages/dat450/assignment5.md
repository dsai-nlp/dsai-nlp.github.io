# DAT450/DIT247: Programming Assignment 5: Retrieval-augmented text generation
In this assignment we will build our own RAG pipeline using LangChain.

## Pedagogical purposes of this assignment
- Get an understanding of how RAG can be used within NLP.
- Learn how to use LangChain to build NLP applications.
- Get an understanding for the challenges and use cases of RAG.

## Requirements
Please submit your solution in Canvas. **Submission deadline: December 8.**

Submit Python files containing your solution to the programming tasks described below. In addition, to save time for the people who grade your submission, please submit a text file containing the outputs printed out by your Python program; read the instructions carefully so that the right outputs are included. 

This is a pure programming assignment and you do not have to write a technical report: there will be a separate individual assignment where you will answer some conceptual questions about what you have been doing here. 

## Step 0: Preliminaries

For the assignments in the course, you can use an environment we have prepared for this course: `/data/courses/2025_dat450_dit247/venvs/dat450_venv`. (So to activate this environment, you type `source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate`.)

If you are running on Colab or your own environment, make sure the following packages are installed:
```bash
pip install langchain 
pip install langchain-community
pip install langchain-huggingface
pip install langchain-core
pip install sentence_transformers
pip install langchain-chroma
```

## Step 1: Get the dataset
You will be working with the [PubMedQA dataset](https://github.com/pubmedqa/pubmedqa) described in this [paper](https://aclanthology.org/D19-1259.pdf). The dataset has been created based on medical research papers from [PubMed](https://pubmed.ncbi.nlm.nih.gov/), you can read more about it in the linked paper.

Use the following code to get the dataset for the assignment. 

If you are running on Minerva or your own environment, run the following command in your command line. Otherwise if you are using notebook e.g. Colab, you can write the following command in a code block with an extra `!` before and run the code block. 

```bash
wget https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json
```

### Collect two datasets
You will collect two datasets from the downloaded file:
- 'questions': the questions with corresponding gold long answer, gold document ID, and year.
- 'documents': the abstracts (contexts+long_answer concatenated), and year.

You can run the following codes to collet these two datasets.

```python
import pandas as pd
tmp_data = pd.read_json("ori_pqal.json").T
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

**Sanity check:** You can print out some of the data in the dataset.

An example of a question our RAG pipeline should answer:
```
questions.iloc[0].question
```

An example of a document the pipeline can leverage to answer the questions:

```
documents.iloc[0].abstract
```


## Step 2: Configure your LangChain LM

### Step 2.1: Find a language model from HuggingFace

Define a language model that will act as the generative model in your RAG pipeline. You can browse for different Hugging Face models on their [webpage](https://huggingface.co/models). 



> Some interesting models (e.g. Llama 3.2) may require that you apply for access. This process is usually quite fast, while it may require that you create an account on Hugging Face (it is free). To use a gated model you need to generate a personal HF token and put it as a secret in your notebook (if using Colab). Make sure that the token has enabled "Read access to contents of all public gated repos you can access". 

<details>
  <summary><b>Hint:</b> How to set up HuggingFace Token when using Minerva</summary>

  If you need to use the huggingface token and you are using Minerva, one way to do it is to add the global parameter in your bash file: `export HF_TOKEM = your_{token}`, and then refer to it in your python code: `hf_token = os.getenv('HF_token')`. Also, to avoid your token being misused, remember to remove the actual token you are using from your submission.
  
</details>


### Step 2.2 Load the language model

You can load the HuggingFace language model using `HuggingFacePipeline.from_model_id`

When calling `HuggingFacePipeline`, set `return_full_text=False` to only return the assistant's response, and call `model.invoke(your_prompt)` to retrieve the text of the output.


**Sanity check:** Prompt your LangChain model and confirm that it returns a reasonable output.

**Include the prompt and the output of this model in your output file.**

## Step 3: Set up the document database

### Step 3.1: Embedding model
First, you need a model to embed the documents in the retrieval corpus. Here, we recommend using the [HuggingFaceEmbeddings](https://docs.langchain.com/oss/python/integrations/text_embedding/huggingfacehub) function.

**Sanity check:** Pass a text passage to the embedding model by calling `embed_query` and evaluate its shape. It should be of the shape (embedding_dim,). 


### Step 3.2: Chunking
Second, you need to chunk the documents in your retrieval corpus, as some likely are too long for the embedding model. Here, you can use the [RecursiveCharacterTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) as a start. The retrieval corpus is given by `documents.abstract`, so you can use `create_documents` on the text splitter with the retrieval corpus to create LangChain `Document` objects, and then use `split_documents` to create text chunks that will be used in creating the vector store.

For evaluation in Step 5, we recommend saving the document id as `metadatas` when creating the document:

```python 
metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(texts=documents.abstract.tolist(), metadata=metadatas)
```

**Sanity check:** Print some samples from the text chunks and check that it makes sense. This way, you might be able to get a feeling for a good chunk size.

### Step 3.3: Define a vector store
Third, you need a vector store to store the documents and corresponding embeddings. There are many document databases and retrievers to play around with. As a start, you can use the [Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/) vector store with cosine similarity as the distance metric. 

When building your vector store, pass the embedding model in [Step 3.1](#step-31-embedding-model) as the embedding model and use the text chunks in [Step 3.2](#step-32-chunking) as the documents in the vector store. To add documents in the vector store, you can Use `Chroma.from_documents` when creating the vector store or use `vector_store.add_documents` after creating the vector store.


**Sanity check:** Query your vector store as follows and check that the results make sense:
```python
results = vector_store.similarity_search_with_score(
    "What is programmed cell death?", k=3
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
```



## Step 4: Define the full RAG pipeline

In this and the following steps, we will guadually build a RAG chain. 

There could be two options of building a RAG chain, and you can choose either **one** of them to build your own RAG: 

[Option A](#option-a-build-a-rag-agent-based-on-the-official-langchain-guide): Build a RAG agent based on the official LangChain guide: [here](https://docs.langchain.com/oss/python/langchain/rag). Here we will use a two-step chain, in which we will run a search in the vector store, and incorporate the result as context for LLM queries.

[Option B](#option-b-build-a-rag-chain-based-on-langchain-open-tutorial): Build a RAG chain using LangChain Expression Language (LCEL) based on a LangChain Open Tutorial: [here](https://colab.research.google.com/github/LangChain-OpenTutorial/LangChain-OpenTutorial/blob/main/13-LangChain-Expression-Language/05-RunnableParallel.ipynb#scrollTo=635d8ebb). Here we will use the [RunnableParallel](https://reference.langchain.com/python/langchain_core/runnables/?h=runnablepara#langchain_core.runnables.base.RunnableParallel) class to build a RAG chain that will also return the retrieved document.

### Option A: Build a RAG agent based on the official LangChain guide

Here, we will define a custom prompt while incorporating the retrieval step.

In order to access the documents retrieved, we can create the prompt in a way that it will [return the source documents](https://docs.langchain.com/oss/python/langchain/rag#returning-source-documents).

```python
from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1] # get the user input query
        retrieved_docs = self.vector_store.similarity_search(last_message.text)  # search for documents

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)  

        augmented_message_content = (
            # Put your prompt here
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }

```

As a start, you might want to fetch only one document per prompt.

<details>
  <summary><b>Hint:</b> Prompt model for classification later</summary>

  In Step 5, we will be using the RAG agent to evaluate whether the model can correctly answer the questions with "Yes" or "No". For evaluation, you may want to prompt the model in a way that it will return only "Yes" or "No" or at least lead the answer with "Yes" or "No".
  
</details>

We are now ready to create a RAG agent. In this step, we can use `create_agent` to build a RAG agent, and use a `RetrieveDocumentsMiddleware` object to act as the middleware.

**Sanity check:** Take a question from your dataset and check whether the model seems to retrieve a relevant document, and answer in a reasonable fashion.

To print out the results prettily, you can use the solution given by Langchain:

```python
for step in agent.stream(
    {"messages": [{"role": "user", "content": your_query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

**Include the prompt and the output of this model in your output file.**

### Option B: Build a RAG chain based on LangChain Open Tutorial

Here, we will firstly define a retriever on the vector store to retrieve documents:

```python
retriever = vectorstore.as_retriever()
```

As a start, you might want the retriever to fetch only one document per prompt.

Then, define your template and use `ChatPromptTemplate.from_template` to create a Chat Prompt. 

With the retriever and the prompt, you should be able to define the RAG chain. In order to return the retrieved context as well as the answers for further evaluation, firstly we can define a `RunnableParallel` object that can take the context and the question, then we can define a chain that only generate text outputs like this:

```python
# Construct the retrieval chain
chain = (
    prompt
    | model
    | StrOutputParser()
)
```
Lastly, combine the `RunnableParallel` object with the chain using the [`assign`](https://reference.langchain.com/python/langchain_core/runnables/?h=runnablepara#langchain_core.runnables.base.RunnableParallel.assign) method. 

```python
rag_chain = runnable_parallel_object.assign(answer=chain)
```

Then you should be able to access the retrieved documents with `answer["context"]`.

**Sanity check:** Take a question from your dataset and check whether the model seems to retrieve a relevant document, and answer in a reasonable fashion.

**Include the prompt and the output of this model in your output file.**

## Step 5: Evaluate RAG on the dataset

Here we will do 4 evaluation tasks to evaluate the RAG agent with the given dataset.

1. Evaluate your full RAG pipeline on the medical questions (`questions.question`) and corresponding gold labels (`questions.gold_label`). 

Since the gold labels can be casted to a binary variable (yes/no) you may use the f1 and/or accuracy metrics.

We expect the model to give answers of "Yes" or "No", but it can happen that the model gives random answers. In this case, one way to perform the evaluation is to keep track of the number of valid answers and do evaluation only on the valid answers.

2. As a baseline, run the same LM without context and compare the performance of the two setups. You can use the same evaluation method as the previous RAG evaluation. Did the retrieval help? 

3. Also evaluate whether the gold documents are fetched for each question. You can compare the retrieved document id with the gold document with ID given by `questions.gold_document_id`.

4. Also, inspect some retrieved documents and corresponding model answers. Does the pipeline seem to work as intended?

**Include the evaluation results in your output file.**