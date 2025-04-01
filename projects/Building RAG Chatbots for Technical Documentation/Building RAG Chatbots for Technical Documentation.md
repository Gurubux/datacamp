Building RAG Chatbots for Technical Documentation
You're working for a well-known car manufacturer who is looking at implementing LLMs into vehicles to provide guidance to drivers. You've been asked to experiment with integrating car manuals with an LLM to create a context-aware chatbot. They hope that this context-aware LLM can be hooked up to a text-to-speech software to read the model's response aloud.

As a proof of concept, you'll integrate several pages from a car manual that contains car warning messages and their meanings and recommended actions. This particular manual, stored as an HTML file, mg-zs-warning-messages.html, is from an MG ZS, a compact SUV. Armed with your newfound knowledge of LLMs and LangChain, you'll implement Retrieval Augmented Generation (RAG) to create the context-aware chatbot.


## Project Instructions
Add your OpenAI API key as an environment variable as described in the notebook.

The car manual HTML document has been loaded for you as car_docs. Using Retrieval Augmented Generation (RAG) to make an LLM of your choice (OpenAI's gpt-4o-mini is recommended) aware of the contents of car_docs, answer the following user query:

"The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"

Store the answer to the user query in the variable answer.


How to approach the project
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