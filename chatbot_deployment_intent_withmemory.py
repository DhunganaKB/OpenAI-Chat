import os
import openai
import streamlit as st

# pip install streamlit-chat
from streamlit_chat import message
from langchain.memory import MongoDBChatMessageHistory

# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(), override=True)

headers = {
    "authorization":st.secrets['OPENAI_API_KEY'],
    "content-type":"application/json",
    "connection_string":st.secrets['CONNECTION_STRING'],
    "pinecone-api-key":st.secrets['PINECONE_API_KEY'],
    "pinecone-env-name":st.secrets['PINECONE_ENV_NAME']
    }
openai.api_key = st.secrets["OPENAI_API_KEY"]

def insert_or_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    #pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV_NAME'))
    pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENV_NAME"])

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        # pinecone.create_index(index_name, dimension=1536, metric='cosine')
        # vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        vector_store=None
        print('Ok')
    return vector_store

index_name='intentdocument-index'
vector_store = insert_or_fetch_embeddings(index_name)

def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history

st.title("ChatBot : INTENT")

chat_history=[]
#question = st.text_input('Enter your question')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

query = st.text_input("Query: ", key="input")

message_history = MongoDBChatMessageHistory(connection_string=st.secrets["CONNECTION_STRING"], session_id="test1")

if query:
    result, chat_history = ask_with_memory(vector_store, question=query, chat_history=chat_history)
    answer = result['answer']
    st.session_state.past.append(query)
    st.session_state.generated.append(answer)
    message_history.add_user_message(result["question"])
    message_history.add_ai_message(result["answer"])
    #print(answer)
    #st.write(answer)
if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
