#Run this cell to install the necessary packages
import subprocess
import pkg_resources

def install_if_needed(package, version):
    '''Function to ensure that the libraries used are consistent to avoid errors.'''
    try:
        pkg = pkg_resources.get_distribution(package)
        if pkg.version != version:
            raise pkg_resources.VersionConflict(pkg, version)
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
        subprocess.check_call(["pip", "install", f"{package}=={version}"])

install_if_needed("langchain-core", "0.3.18")
install_if_needed("langchain-openai", "0.2.8")
install_if_needed("langchain-community", "0.3.7")
install_if_needed("unstructured", "0.14.4")
install_if_needed("langchain-chroma", "0.1.4")
install_if_needed("langchain-text-splitters", "0.3.2")


# Set your API key to a variable
import os
openai_api_key = os.environ["OPENAI_API_KEY"]

# Import the required packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# Load the HTML as a LangChain document loader
loader = UnstructuredHTMLLoader(file_path="data/mg-zs-warning-messages.html")
car_docs = loader.load()


# Step 1: Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
car_doc_chunks = text_splitter.split_documents(car_docs)

# Step 2: Embed and store the document chunks
embedding = OpenAIEmbeddings(api_key=openai_api_key)
vectorstore = Chroma.from_documents(documents=car_doc_chunks, embedding=embedding)

# Step 3: Create a retriever
retriever = vectorstore.as_retriever()

# Step 4: Initialize the LLM and prompt template
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

# Step 5: Define the RAG chain
from langchain.chains import RetrievalQA

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Step 6: Invoke the RAG chain with the query
query = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"
response = rag_chain.invoke(query)
answer = response.content  # Extract the string content from the AIMessage

# Display the answer
print(answer)
