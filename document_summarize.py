import os
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(), override=True)

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
import streamlit as st
from tempfile import NamedTemporaryFile
from PyPDF2 import PdfFileReader

headers = {
    "authorization":st.secrets['openai_api_token'],
    "content-type":"application/json"
    }

st.title('Text Summarization')
#st.write('upload pdf file for summarization')
original_text = '<p style="font-family:Courier; color:Blue; font-size: 20px;">upload pdf file to extract summary</p>'
st.markdown(original_text, unsafe_allow_html=True)

## Reading input mp4 file and converting it into .mp3 file
uploaded_file = st.sidebar.file_uploader("Upload your pdf file", type="pdf")

def processing(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

def get_summary(chunks):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k')
    prompt_template = """Write a concise summary of the following extracting the key information:
    Text: `{text}`
    CONCISE SUMMARY:"""
    initial_prompt = PromptTemplate(template=prompt_template, input_variables=['text'])
    refine_template = '''
        Your job is to produce a final summary.
        I have provided an existing summary up to a certain point: {existing_answer}.
        Please refine the existing summary with some more context below.
        ------------
        {text}
        ------------
        Start the final summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
        by BULLET POINTS AND end the summary with a CONCLUSION PHRASE.
        
    '''
    refine_prompt = PromptTemplate(
        template=refine_template,
        input_variables=['existing_answer', 'text']
    )
    chain = load_summarize_chain(
        llm=llm,
        chain_type='refine',
        question_prompt=initial_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False,
    )
    output_summary = chain.run(chunks)
    return output_summary


if uploaded_file is not None:
    st.write('Waiting time depends upon the size of the pdf file.')
    temp_file = NamedTemporaryFile(delete=False)
    temp_path = temp_file.name
    temp_file.close()

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Use the saved file path with UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(temp_path)
    data = loader.load()

    # Delete the temporary file
    os.remove(temp_path)

    # my_list = []
    # mm = 0
    with st.spinner('Model working ...'):
        chunks = processing(data)
        output=get_summary(chunks)
    st.subheader('Summary Report')
    st.write(output)

