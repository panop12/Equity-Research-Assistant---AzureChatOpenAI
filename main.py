import langchain
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
import os


def generate_equity_research_prompt(user_question, context=None):
    template_str = """
    You are an expert financial analyst specializing in equity research. Provide a detailed analysis based on the provided context and question.

    Context: {context}

    Question: {question}

    Ensure your response covers the following aspects:
    Make sure that the format is easy to read 
    - Analyze and explain SET today market and condition/strategy , tell support and resistance
    - Summarize today hightlights
    - Analyze and suggest today trading strategy 
    - Suggest stock to invest in today from daily top picks
    - Provide backgroud infomation about those stock and why to invest
    - Recent market trends affecting the stock 
    - Competitive landscape
    - Potential risks and opportunities
    """
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=template_str
    )

    context = context or "No additional context provided."
    prompt = prompt_template.format(question=user_question, context=context)

    return prompt
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks,api_key, api_base, api_version, model):
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        openai_api_base=api_base,
        openai_api_version=api_version,
        model=model,
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore,api_key, api_base, api_version, model):
    llm = AzureChatOpenAI(
        deployment_name=model,
        openai_api_key=api_key,
        openai_api_base=api_base,
        openai_api_version=api_version,
        temperature=0.0,
        max_tokens=7000,
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    context = "Provide any necessary context related to the equity or company in question, such as recent news, financial reports, or market conditions."
    prompt = generate_equity_research_prompt(user_question, context)

    if st.session_state.chat_history == None :
         response = st.session_state.conversation({'question': prompt})
    else:
         response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
    OPENAI_API_DEPLOYMENT_NAME = os.getenv("OPENAI_API_DEPLOYMENT_NAME")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
    OPENAI_EMBEDDING_SIZE = os.getenv("OPENAI_EMBEDDING_SIZE")

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("INVX research assistant")
    user_question = st.text_input("Ask a question about your pdf:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks,OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION, OPENAI_EMBEDDING_MODEL)
                st.session_state.conversation = get_conversation_chain(vectorstore,OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION, OPENAI_API_DEPLOYMENT_NAME)
                langchain.debug = True

if __name__ == '__main__':
    main()