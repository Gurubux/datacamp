# Building RAG Chatbots for Technical Documentation
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