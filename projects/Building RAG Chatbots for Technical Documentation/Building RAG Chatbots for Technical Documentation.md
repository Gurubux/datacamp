# Building RAG Chatbots for Technical Documentation
Implement retrieval augmented generation (RAG) with LangChain to create a chatbot for answering questions about technical documentation.
Project Description
You'll create a context-aware chatbot by integrating a car manual with an LLM using LangChain and Retrieval Augmented Generation (RAG). The goal is to create a car assistant that can explain dashboard warnings and recommend actions while driving. Say goodbye to boring manuals!


	You're working for a well-known car manufacturer who is looking at implementing LLMs into vehicles to provide guidance to drivers. You've been asked to experiment with integrating car manuals with an LLM to create a context-aware chatbot. They hope that this context-aware LLM can be hooked up to a text-to-speech software to read the model's response aloud.
	As a proof of concept, you'll integrate several pages from a car manual that contains car warning messages and their meanings and recommended actions. This particular manual, stored as an HTML file, mg-zs-warning-messages.html, is from an MG ZS, a compact SUV. Armed with your newfound knowledge of LLMs and LangChain, you'll implement Retrieval Augmented Generation (RAG) to create the context-aware chatbot.


## Project Instructions
	Add your OpenAI API key as an environment variable as described in the notebook.
	The car manual HTML document has been loaded for you as car_docs. Using Retrieval Augmented Generation (RAG) to make an LLM of your choice (OpenAI's gpt-4o-mini is recommended) aware of the contents of car_docs, answer the following user query:
	"The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"
		Store the answer to the user query in the variable answer.


## How to approach the project
1. Split the document
	Split the HTML document into chunks.
	 - Initializing a splitter
	 - Splitting the text

2. Store the embeddings
	Embed and store the document chunks for retrieval.
	- Where to store the embeddings
	- Storing embeddings in a Chroma vector database

3. Create a retriever
	Create a retriever to retrieve relevant documents from the vector store.

4. Initialize the LLM and prompt template
	Define an LLM and create a prompt template to set up the RAG workflow.
	- Initialize LLM
	- Define a chat prompt template

5. Define RAG chain
	Define RAG chain to connect the retriever, question, prompt, and LLM.
	Defining the RAG chain using LangChain Expression Language (LCEL)

6. Invoke RAG chain
	Invoke your chain with the user query to answer.



Sure! Here's a detailed explanation of the 6 steps involved in building a RAG chatbot for car manuals using LangChain, written in Markdown (`.md`) format:

---

## Building a RAG Chatbot for Technical Documentation

This guide walks through how to build a Retrieval Augmented Generation (RAG) chatbot using LangChain, OpenAI, and ChromaDB. The chatbot is capable of understanding and responding to queries based on a car manual loaded from an HTML file.

---

### Step 1: Split the Document

**Library Used:**
- `langchain_text_splitters.RecursiveCharacterTextSplitter`

**What it does:**

Loads the HTML file using a document loader, then splits the long document into smaller, overlapping text chunks so that the LLM can handle them more effectively. This helps ensure contextual completeness during retrieval.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
car_doc_chunks = text_splitter.split_documents(car_docs)
```

---

### Step 2: Store the Embeddings

**Libraries Used:**
- `langchain_openai.OpenAIEmbeddings`
- `langchain_chroma.Chroma`

**What it does:**

Generates vector embeddings (numerical representations) for each chunk of the car manual and stores them in a Chroma vector database for efficient retrieval later.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding = OpenAIEmbeddings(api_key=openai_api_key)
vectorstore = Chroma.from_documents(documents=car_doc_chunks, embedding=embedding)
```

---

### Step 3: Create a Retriever

**Library Used:**
- `Chroma.as_retriever()`

**What it does:**

Creates a retriever from the vector store. When the user asks a question, the retriever searches for the most relevant chunks of the manual using similarity search.

```python
retriever = vectorstore.as_retriever()
```

---

### Step 4: Initialize the LLM and Prompt Template

**Libraries Used:**
- `langchain_openai.ChatOpenAI`
- `langchain_core.prompts.ChatPromptTemplate`

**What it does:**

Initializes the LLM (OpenAI GPT-4o-mini is recommended) and sets up a prompt template that guides the LLM to answer the user’s query using only the retrieved context.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

prompt = ChatPromptTemplate.from_template("""
You are an expert assistant for a car manual. Use the following context to answer the question at the end.
If the answer is not in the provided context, say you don’t know and suggest checking the official manual.

Context:
{context}

Question:
{question}
""")
```

---

### Step 5: Define the RAG Chain

**Libraries Used:**
- `langchain_core.runnables.RunnablePassthrough`
- `LangChain Expression Language (LCEL)`

**What it does:**

Connects the retriever, prompt template, and the LLM into a single chain. This RAG chain takes a query, retrieves relevant documents, formats them into a prompt, and returns a generated answer.

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

---

### Step 6: Invoke the RAG Chain

**What it does:**

Sends a user query through the RAG chain and extracts the LLM’s response as a plain string, storing it in the `answer` variable.

```python
query = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"
response = rag_chain.invoke(query)
answer = response.content  # Extract string content
```


## QUESTION
from langchain.chains import RetrievalQA 

Was this required?

Great question — and **no**, in your current implementation using **LangChain Expression Language (LCEL)** with this line:

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

➡️ **`from langchain.chains import RetrievalQA` is *not* required.**

---

### 🧠 When would you use `RetrievalQA`?

You would import and use `RetrievalQA` if you were following the older or alternate way of constructing a RAG pipeline like this:

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
```

But since you're using **LCEL**, which is more flexible and composable, you can skip `RetrievalQA`.

✅ So, feel free to remove this line:
```python
from langchain.chains import RetrievalQA
```

It's unused in your current setup.



Great questions again — let’s walk through **Step 2** and break everything down so it’s crystal clear.

---

## 🧠 Step 2: Embed and Store the Document Chunks

```python
embedding = OpenAIEmbeddings(api_key=openai_api_key)
vectorstore = Chroma.from_documents(documents=car_doc_chunks, embedding=embedding)
```

---

### 🔹 What is an **Embedding**?

An **embedding** is a way of turning text into numbers — specifically, into a **dense vector** of real numbers — so a computer can understand its **semantic meaning**.

For example:

```python
Text: "engine warning"
Embedding: [0.31, -0.45, 0.22, ..., 0.01]  # A list of 1536 numbers (OpenAI's default)
```

These embeddings capture **semantic similarity**. For instance:

- "engine alert" and "engine warning" → very close vectors ✅
- "windshield washer fluid low" → further apart ❌

---

### 🔹 Code Breakdown: OpenAI Embeddings

```python
embedding = OpenAIEmbeddings(api_key=openai_api_key)
```

This uses OpenAI’s `text-embedding-3-small` (or default model) to convert **each chunk** of the manual into a vector. Each vector has **1536 dimensions**.

---

### 🔸 Are there other embedding models?

Yes! Some common ones:

| Embedding Model | Provider        | Pros                                           | Cons                                           |
|------------------|------------------|------------------------------------------------|------------------------------------------------|
| `OpenAIEmbeddings` | OpenAI         | High quality, great for general tasks          | Paid, requires API key                         |
| `HuggingFaceEmbeddings` | HuggingFace   | Many free models (e.g., `sentence-transformers`) | May require setup, GPU for speed               |
| `SentenceTransformers` | SBERT         | Great semantic embeddings                     | Needs installation and compute                 |
| `CohereEmbeddings` | Cohere         | High quality, commercial ready                | API pricing                                   |
| `Google Vertex AI Embeddings` | Google Cloud | Useful if you're in GCP ecosystem             | Setup-heavy                                    |

✅ **When to choose which?**
- For **proof of concept or high performance**: Use OpenAI
- For **local or open-source setup**: Use `sentence-transformers`
- For **GCP/Azure/AWS integrations**: Use respective cloud embeddings

---

### 🔹 What is **Chroma**?

Chroma is an open-source **vector database**. It allows:
- Storing text chunks and their embeddings
- Performing **semantic search** over them (find similar texts by meaning)

Think of it as a smart search engine that indexes meaning, not just words.

---

### 🔸 Code Breakdown: Storing in Chroma

```python
vectorstore = Chroma.from_documents(documents=car_doc_chunks, embedding=embedding)
```

- `car_doc_chunks` = list of LangChain `Document` objects
- `embedding` = how to convert each document into a vector
- `Chroma.from_documents()`:
  - Converts each document into an embedding
  - Stores the text and vector in a mini in-memory or persistent DB

---

### 🔸 What does the **vectorstore** look like?

Internally, it’s a searchable collection of:

| ID       | Text Chunk                                  | Vector (Embedding)                        |
|----------|----------------------------------------------|-------------------------------------------|
| doc_001  | "If the engine light turns red, stop car..." | [0.31, -0.45, 0.22, ..., 0.01]            |
| doc_002  | "This light means tire pressure is low..."   | [0.29, -0.50, 0.19, ..., 0.02]            |

Each entry links a text chunk with its numeric representation.

---

### 🔸 How is it stored?

By default:
- Stored **in-memory** during development
- Can be stored **on-disk** with a folder like:

```python
Chroma.from_documents(..., persist_directory="chroma_db")
```

To persist the data and reuse it later, call:

```python
vectorstore.persist()
```

✅ You can then reload it with:

```python
Chroma(persist_directory="chroma_db", embedding_function=embedding)
```

---

## ✅ Summary

| Concept         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Embedding**     | Turns text into a numerical vector to capture meaning                     |
| **OpenAIEmbeddings** | Converts text using OpenAI’s embedding model                             |
| **Chroma**        | A vector database that stores text chunks + their embeddings               |
| **vectorstore**   | A retrievable memory of documents and their semantic meanings              |
| **Storage**       | By default in RAM, but can be persisted to disk for production use         |

---


Great! Let’s unpack exactly what happens in **Step 3** with this line:

```python
retriever = vectorstore.as_retriever()
```

---

## 🔹 What is a Retriever?

A **retriever** is a LangChain abstraction that enables **semantic search** over your vector database.

You give it a **user question**, and it returns the **most relevant document chunks** based on their **vector similarity** to the query.

---

## 🔹 What does `vectorstore.as_retriever()` do?

```python
retriever = vectorstore.as_retriever()
```

This converts the Chroma vector store into a **Retriever** object, which can:

- Accept a user query
- Embed the query using the same embedding function used during storage
- Find the most similar chunks in the vector database
- Return them as `Document` objects (with `page_content` and `metadata`)

This is the **core retrieval** step in Retrieval-Augmented Generation (RAG).

---

## 🔹 What happens internally?

Here's what happens under the hood:

1. ✅ **User Query Input**  
   You pass in a query string like:
   ```python
   retriever.invoke("What does the tire pressure warning light mean?")
   ```

2. 🔄 **Embedding the Query**  
   The retriever uses the same embedding model (e.g., OpenAI) to convert this string into a vector.

3. 🧠 **Vector Similarity Search**  
   It compares the query vector to all the vectors in your `vectorstore` using cosine similarity or similar metrics.

4. 📚 **Returns Relevant Chunks**  
   It returns the top N most relevant `Document` chunks (default is 4).

---

## 🔹 Can you customize it?

Yes! You can tune how the retriever works:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",   # or "mmr" (Max Marginal Relevance)
    search_kwargs={"k": 3}      # Top-k most relevant chunks
)
```

### Common options:
| Parameter         | Description                                        |
|-------------------|----------------------------------------------------|
| `search_type`     | `"similarity"` (default) or `"mmr"` (for diversity) |
| `search_kwargs`   | e.g., `{ "k": 5 }` to return top 5 documents       |

---

## ✅ Summary

| Concept            | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `as_retriever()`     | Wraps the vectorstore as a retriever object for semantic search            |
| Returns              | Most relevant document chunks for a given query                           |
| How it works         | Embeds the query → computes similarity → returns matching chunks           |
| Customizable?        | Yes (`k`, `search_type`, etc.)                                             |

---

Excellent question — and it's a subtle but important distinction!

Let’s break it down clearly:

---

## ⚔️ `retriever.invoke("your query")` vs `retriever.get_relevant_documents("your query")`

| Function                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `retriever.get_relevant_documents(query)` | **Low-level method** that returns a list of relevant `Document` objects     |
| `retriever.invoke(query)`       | **High-level interface** used in LangChain Expression Language (LCEL); internally calls `get_relevant_documents` |

---

### 🧠 In simple terms:

- **Use `get_relevant_documents()`** when you're manually writing or debugging retrieval logic and want full control.
- **Use `invoke()`** when working with **LCEL chains** (LangChain’s pipe `|` syntax) — it's designed to play nicely with the rest of the RAG pipeline.

---

### ✅ Example 1: Manual retrieval

```python
docs = retriever.get_relevant_documents("What does the oil light mean?")
for doc in docs:
    print(doc.page_content)
```

> Returns a **list of Document objects**, each containing a relevant chunk of text.

---

### ✅ Example 2: Using `invoke()` in LCEL chain

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

response = rag_chain.invoke("What does the oil light mean?")
```

> Here, `retriever.invoke(...)` is **automatically called** when used in the context of LCEL, thanks to LangChain’s dynamic routing of components.

---

### 🧪 TL;DR Summary

| Use Case | Method                   | Returns                     |
|----------|--------------------------|-----------------------------|
| Manual access | `get_relevant_documents(query)` | `List[Document]`               |
| Inside LCEL chain | `invoke(query)` (automatically)  | Input for the next chain step |

---

Great set of questions — this step is where the **magic happens** by bringing the LLM into the RAG pipeline. Let’s break it all down carefully 👇

---

## ✅ Step 4: Initialize the LLM and Prompt Template

```python
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

prompt = ChatPromptTemplate.from_template(
    """
    You are an expert assistant for a car manual. Use the following context to answer the question at the end.
    If the answer is not in the provided context, say you don’t know and suggest checking the official manual.

    Context:
    {context}

    Question:
    {question}
    """
)
```

---

## 🧠 `ChatOpenAI` — What is it?

This is a **LangChain wrapper** around OpenAI's **chat-based models**, such as:
- `gpt-3.5-turbo`
- `gpt-4`
- `gpt-4-turbo`
- `gpt-4o-mini` ✅ (used here)

It allows you to plug LLMs into chains with a simple interface.

### 🔹 Common Parameters

| Parameter         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `model`          | Name of the OpenAI model (e.g., `gpt-4`, `gpt-4o-mini`, `gpt-3.5-turbo`)     |
| `api_key`        | Your OpenAI API key                                                          |
| `temperature`    | Controls creativity (0 = factual, 1 = more creative)                         |
| `max_tokens`     | Max length of the model's output                                             |
| `streaming`      | Stream tokens back gradually if `True`                                       |
| `timeout`        | Optional timeout for API requests                                            |

📌 **Example**:
```python
llm = ChatOpenAI(model="gpt-4", temperature=0.3, max_tokens=500)
```

---

## 💬 `ChatPromptTemplate` — What is it?

`ChatPromptTemplate` is LangChain’s way of building **structured chat prompts** for use with chat-based models (like GPT-4). It’s cleaner and more composable than raw string concatenation.

### 🔹 `from_template(...)` — What does it do?

This creates a template where **placeholders** like `{context}` and `{question}` will be dynamically **filled in at runtime**.

So if the user later asks:
> _"What does the engine warning light mean?"_

and the retriever returns:
> _"This light indicates engine malfunction; stop the car immediately."_

Then the full prompt becomes:

```
You are an expert assistant for a car manual. Use the following context to answer the question at the end.
If the answer is not in the provided context, say you don’t know and suggest checking the official manual.

Context:
This light indicates engine malfunction; stop the car immediately.

Question:
What does the engine warning light mean?
```

### 📌 Is `prompt` just a string?

No — it’s a **template object** that dynamically renders a string **at runtime** when passed input variables (`context`, `question`).

---

## 🔄 Other Prompt Options

LangChain supports multiple prompt classes:

| Prompt Type           | Use Case                                           | Notes                                                   |
|-----------------------|----------------------------------------------------|----------------------------------------------------------|
| `ChatPromptTemplate`  | For chat models like GPT-4                         | Supports system/user message roles                       |
| `PromptTemplate`      | For completion models (e.g., `text-davinci-003`)   | Single block of text, not structured as chat             |
| `FewShotPromptTemplate` | For in-context examples                        | You define a few Q&A examples + input                    |
| `MessagesPlaceholder` | For memory (chat history) in multi-turn chats      | Plug into `ChatPromptTemplate`                          |

---

## 📦 `context` and `question` — What are they?

These are **variables** in your template.

- `context`: Filled with relevant document chunks returned by the retriever
- `question`: Filled with the user’s input query

So when the RAG pipeline runs:

```python
prompt.format(context=relevant_text, question=user_input)
```

…it creates the full input prompt for the LLM.

---

## ⚙️ What happens under the hood?

When the prompt is passed to the chain:

1. **`ChatPromptTemplate` renders the prompt** using retrieved context and the user’s question.
2. The final string is passed to `ChatOpenAI`, which calls OpenAI's API and returns the response.
3. The output is a `ChatMessage` object, e.g. `AIMessage(content="...")`.

---

## ✅ Summary Table

| Concept                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `ChatOpenAI`           | LLM interface for OpenAI chat models                                        |
| `ChatPromptTemplate`   | Template builder for chat prompts with variables like `{context}`           |
| `from_template()`      | Quickly builds a template from a string with placeholders                   |
| `context`              | Inserted text from retriever (document chunks)                              |
| `question`             | User’s natural language query                                                |
| `prompt`               | A **template object**, not a plain string                                   |
| Output of LLM          | A `ChatMessage` with `.content` containing the generated answer             |

---

Excellent — **Step 5** is where you build the complete **RAG (Retrieval-Augmented Generation) chain**, and yes, this example uses the newer and very powerful **LangChain Expression Language (LCEL)**.

Let’s break this down step by step 👇

---

## 🧠 What is happening here?

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

This defines a **pipeline** where:

1. A dictionary is created:
   - `"context"` → output of the `retriever` (retrieves relevant docs)
   - `"question"` → passed through unchanged using `RunnablePassthrough()`
2. That dictionary is passed into the `prompt` template
3. The formatted prompt is passed into the `llm`
4. The final output is generated by the LLM

✅ This is a full RAG pipeline built with **LCEL**.

---

## 🔍 What is LCEL (LangChain Expression Language)?

> LCEL is a declarative, functional syntax introduced by LangChain that lets you **build chains** using the `|` (pipe) operator.

Think of it like Unix pipes or function composition.

### ✅ Benefits of LCEL:

| Feature               | Benefit                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| **Composable**        | Easily chain components (retriever → prompt → LLM)                      |
| **Readable**          | Minimal boilerplate, linear flow                                        |
| **Flexible**          | Supports conditional logic, branching, parallel execution               |
| **Faster**            | Runs with LangChain's compiled engine, making it more efficient         |
| **Tooling-friendly**  | Integrates with tracing, testing, caching, observability tools          |

---

### 🔸 Before LCEL — the old way

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
```

- ✅ Still works
- ❌ Less transparent and harder to customize (especially prompt & flow)
- ❌ More boilerplate

---

## 🔍 Explanation of LCEL syntax

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

### Breakdown:

| Component                     | Purpose                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| `{"context": retriever}`     | Retrieves relevant docs and injects them into the `{context}` placeholder |
| `"question": RunnablePassthrough()` | Just forwards the user’s input as `{question}`                            |
| `| prompt`                   | Formats the input using your `ChatPromptTemplate`                      |
| `| llm`                      | Sends the formatted prompt to OpenAI and gets a response               |

---

## ⚙️ `RunnablePassthrough()` — What is that?

This is a utility that **just passes the input as-is**. So when a user sends a query:

```python
"What's the engine warning light?"
```

…it becomes:
```python
{"question": "What's the engine warning light?"}
```

---

## 🔧 Other Parameters / Flexibility in LCEL

You can customize almost everything:

### ✨ Control how many docs to retrieve:
```python
retriever.search_kwargs["k"] = 5
```

### ✨ Add a custom function in the chain:
```python
from langchain_core.runnables import RunnableLambda

clean = RunnableLambda(lambda x: x.strip())

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | clean
    | llm
)
```

### ✨ Add conditional logic:
```python
from langchain_core.runnables import RunnableBranch

chain = RunnableBranch(
    (lambda x: "error" in x, error_chain),
    default_chain
)
```

### ✨ Parallel execution:
```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel({
    "docs": retriever,
    "question": RunnablePassthrough()
})
```

---

## ✅ Summary

| Concept                        | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| **LCEL**                       | LangChain Expression Language — a pipe-based way to build chains           |
| **`RunnablePassthrough()`**    | Forwards input unchanged to next step (used for question input)            |
| **`|` operator**               | Chains components together (retriever → prompt → llm)                      |
| **Better than `RetrievalQA`** | Yes, because it's more customizable, readable, and performant              |
| **Other LCEL features**        | Conditional logic, parallel branches, lambda transforms, caching, tracing  |

---

You're at the final step! Let's fully unpack what’s happening in **Step 6** — this is where everything you've built comes together to actually generate a response using your RAG chain.

---

## 🧠 What is happening here?

```python
query = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"
response = rag_chain.invoke(query)
answer = response.content
```

### ✅ Step-by-step breakdown:

1. **User query** is stored in the variable `query` — a natural language question.

2. `rag_chain.invoke(query)` triggers the full RAG pipeline you built using LCEL:
   - 🔍 The **retriever** searches for relevant chunks in the vectorstore using semantic similarity.
   - 📄 The **prompt** is filled with:
     - `context` = retrieved document chunks
     - `question` = the user’s query
   - 💬 The **LLM** receives the final prompt and generates a response.

3. `response` is an **AIMessage object** containing metadata and the response content.

4. `answer = response.content` extracts the final **plain string** reply from the LLM — this is what you'd return to the user or feed into text-to-speech, etc.

---

## ⚙️ What does `.invoke()` do under the hood?

`invoke()` is a **high-level execution method** in LangChain used by all LCEL `Runnable` objects (retrievers, prompts, chains, LLMs).

Under the hood, `.invoke()`:

- Runs **input validation**
- Triggers each component of the chain in sequence
- Handles:
  - Type conversion (str → dict)
  - Prompt formatting
  - API calling (OpenAI)
  - Output wrapping (as `ChatMessage`, etc.)

It’s **synchronous**, so it blocks until the full result is available.

---

## 🔁 Other ways to do this

### 1. **Streaming version**: `stream()`

If you want to stream the LLM's output (e.g., token by token), you can use:

```python
for chunk in rag_chain.stream(query):
    print(chunk.content, end="")
```

Useful for chatbots, UI apps, or real-time TTS.

---

### 2. **Batch processing**: `batch()`

To run multiple queries at once:

```python
queries = ["What does the brake warning mean?", "How to reset the tire pressure sensor?"]
responses = rag_chain.batch(queries)

answers = [r.content for r in responses]
```

Efficient for bulk queries or evaluations.

---

### 3. **Asynchronous version**: `ainvoke()`

For async codebases (e.g., FastAPI):

```python
response = await rag_chain.ainvoke(query)
answer = response.content
```

---

## ✅ Summary

| Line                             | What it does                                                        |
|----------------------------------|---------------------------------------------------------------------|
| `rag_chain.invoke(query)`        | Executes the full RAG pipeline with input query                     |
| `retriever`                      | Finds relevant document chunks                                      |
| `prompt`                         | Fills in the query + context into your defined template             |
| `llm`                            | Generates the final answer using OpenAI                             |
| `response.content`               | Extracts the plain string answer from the model                     |

| Method       | Use Case                        | Sync/Async |
|--------------|----------------------------------|------------|
| `invoke()`   | Standard single query execution | Sync       |
| `stream()`   | Real-time streaming output      | Sync       |
| `batch()`    | Process multiple inputs         | Sync       |
| `ainvoke()`  | Async version of `invoke()`     | Async      |

---

Let me know if you want to wrap this in a chatbot UI, add TTS, or create an evaluation script!