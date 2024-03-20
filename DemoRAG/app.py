import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import streamlit as st


# OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"
loader = Docx2txtLoader("Leave Policy.docx")
data = loader.load()

#documents = data[0].page_content
docuemnt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 50, length_function = len)
chunks = docuemnt_splitter.split_documents(data)
embeddings = OpenAIEmbeddings()
list_chunks=[x.page_content for x in chunks]
# initialize the faiss retriever and faiss retriever
faiss_vectorstore = FAISS.from_texts(list_chunks, embeddings)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})

# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_texts(list_chunks)
bm25_retriever.k = 4

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.5, 0.5])


prompt = ChatPromptTemplate.from_template(
    """ You are a human resource chabot and
    answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

chain_ensemble = (
    RunnablePassthrough.assign(context=(lambda x: x["question"]) | ensemble_retriever)
    | prompt
    | ChatOpenAI(model="gpt-4-1106-preview")
    | StrOutputParser()
)

st.title("ChatBot : Leave Policy")
query = st.text_input("Query: ", key="input")

# query='How many months one should work to be eligible for leave?'
# chain_ensemble.invoke({'question':query})

if query:
    answer=chain_ensemble.invoke({'question':query})
    st.write(answer)

#query='How many months one should work to be eligible for leave?'
#chain_ensemble.invoke({'question':query})
