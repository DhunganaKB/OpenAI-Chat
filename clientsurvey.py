import streamlit as st
from pymongo import MongoClient
from streamlit_chat import message
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

import yagmail
from email.message import EmailMessage
import smtplib

## This is online version, the local copy is older version
headers = {
    "authorization":st.secrets['OPENAI_API_KEY'],
    "content-type":"application/json",
    "connection_string":st.secrets['CONNECTION_STRING'],
    "pinecone-api-key":st.secrets['PINECONE_API_KEY'],
    "pinecone-env-name":st.secrets['PINECONE_ENV_NAME']
    }
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title('Pilot Program for Neste')
CONNECTION_STRING = st.secrets["CONNECTION_STRING"]

embeddings = OpenAIEmbeddings()   
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENV_NAME"]) 

#index_name='intentdocument-index'
index_name='pilotsurvey'
vector_store = Pinecone.from_existing_index(index_name, embeddings)

def get_mongodb_client():
    client = MongoClient(CONNECTION_STRING)
    return client

client = get_mongodb_client()
db = client['chatbot_db']
conversation_collection = db['conversations']

def ask_with_memory(vector_store, user_name, user_email, question, conversation_collection,  chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    conversation_collection.insert_one({
        'user_name':user_name,
        'user_email':user_email,
        'question':question,
        'answer':result['answer']
    })
    return result, chat_history

chat_history=[]

st.write("Please Provide your name and email adress:")
user_name = st.text_input("Your Name:")
user_email = st.text_input("Your Email:")

if user_name and user_email:
    question = st.text_input("if you have any questions about this program, write your questions: ", key="input")

    if question:
        result, chat_history = ask_with_memory(vector_store, user_name, user_email, question, conversation_collection, chat_history=chat_history)

        # message_history.add_user_message(result["question"])
        # message_history.add_ai_message(result["answer"])
        st.write(result['answer'])

    with st.form(key='myform', clear_on_submit=True):
        message = st.text_area("Will you be interested in participing in this program? Please answer yes or no.", height=10)
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            conversation_collection.insert_one({
                'user_name':user_name,
                'user_email':user_email,
                'question':'are you interested in this program?',
                'answer':message})
            if 'yes' in message.lower():
                msg = EmailMessage()
                contacts = [user_email]
                msg['Subject'] = 'Invitation From INTENT'
                msg['From'] = 'dhunganain23@gmail.com'
                msg['To'] = ', '.join(contacts)
                
                msg.set_content(f'''\n Dear {user_name},\n\n Hope this email find you well. We would like to invite you to participate in the sustaibale agricuture pilot program. Thank you for showing you interest. Please find the link below to proceed a head: https://intent.ag/ .
                \n
                Thanks\n
                INTENT
                \n''')
                
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login('dhunganain23@gmail.com', 'dbszoxlqycoidmyi')
                    smtp.send_message(msg)
                st.write(f"We have sent you an email with additional information at {user_email}, Please kindly check your email and let us know if you have any other questions")  
            st.write('Thank you for participating this survey')


                    
                
        




