import streamlit as st
from pymongo import MongoClient
from streamlit_chat import message
from langchain.memory import MongoDBChatMessageHistory
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

headers = {
    "authorization":st.secrets['OPENAI_API_KEY'],
    "content-type":"application/json",
    "connection_string":st.secrets['CONNECTION_STRING'],
    "pinecone-api-key":st.secrets['PINECONE_API_KEY'],
    "pinecone-env-name":st.secrets['PINECONE_ENV_NAME']
    }
openai.api_key = st.secrets["OPENAI_API_KEY"]

CONNECTION_STRING = st.secrets["CONNECTION_STRING"]

print(CONNECTION_STRING)

st.title('Simple question answer chatbot')
   
embeddings = OpenAIEmbeddings()    
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENV_NAME"])

index_name='intentdocument-index'
vector_store = Pinecone.from_existing_index(index_name, embeddings)

def get_mongodb_client():
    client = MongoClient(CONNECTION_STRING)
    return client

def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    # client = get_mongodb_client()
    # db = client['chatbot_db']
    # conversation_collection = db['conversations']
    # conversation_collection.insert_one({
    #     'question': question,
    #     'answer': result['answer']
    # })
    return result, chat_history

chat_history=[]
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
query = st.text_input("Query: ", key="input")

if query:
    result, chat_history = ask_with_memory(vector_store, question=query, chat_history=chat_history)
    answer = result['answer']
    st.session_state.past.append(query)
    st.session_state.generated.append(answer)
    st.write(answer)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
